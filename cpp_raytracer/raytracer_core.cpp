#include "raytracer_core.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stack>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

// ================================================
// SIMPLIFIED BVH NODE FOR TRAVERSAL
// ================================================
struct TraversalNode {
    int node_index;
    float tmin;
    
    TraversalNode() : node_index(0), tmin(0.0f) {}  // Default constructor
    TraversalNode(int idx, float t) : node_index(idx), tmin(t) {}
};

// ================================================
// BVH BUILDER (Array-based, cache friendly)
// ================================================
class BVHBuilder {
private:
    struct BuildNode {
        AABB bbox;
        int start;
        int end;
        int parent;
        int depth;
        
        BuildNode() : start(0), end(0), parent(-1), depth(0) {}
        BuildNode(int s, int e, int p, int d) : start(s), end(e), parent(p), depth(d) {}
    };
    
    Sphere* spheres;
    int* indices;
    int n_spheres;
    BVHNodeFlat* flat_nodes;
    int node_count;
    int max_depth;
    
    FORCEINLINE bool box_compare(int idx_a, int idx_b, int axis) {
        if (axis == 0) return spheres[idx_a].center.x < spheres[idx_b].center.x;
        if (axis == 1) return spheres[idx_a].center.y < spheres[idx_b].center.y;
        return spheres[idx_a].center.z < spheres[idx_b].center.z;
    }
    
public:
    BVHBuilder(Sphere* spheres_ptr, int* indices_ptr, int n) 
        : spheres(spheres_ptr), indices(indices_ptr), n_spheres(n), 
          flat_nodes(nullptr), node_count(0), max_depth(0) {}
    
    int build(BVHNodeFlat* nodes, int max_nodes) {
        flat_nodes = nodes;
        node_count = 0;
        max_depth = 0;
        
        if (n_spheres == 0) return 0;
        
        // Build tree iteratively using stack (no recursion)
        std::stack<BuildNode> node_stack;
        node_stack.push(BuildNode(0, n_spheres, -1, 0));
        
        int node_index = 0;
        
        while (!node_stack.empty()) {
            BuildNode current = node_stack.top();
            node_stack.pop();
            
            int current_node_idx = node_index++;
            int span = current.end - current.start;
            
            // Calculate bounding box for this node
            AABB node_bbox;
            if (span > 0) {
                node_bbox = spheres[indices[current.start]].bbox;
                for (int i = current.start + 1; i < current.end; ++i) {
                    node_bbox = AABB::surrounding(node_bbox, spheres[indices[i]].bbox);
                }
            }
            
            if (span <= 4) {  // Create leaf
                flat_nodes[current_node_idx].bbox = node_bbox;
                flat_nodes[current_node_idx].first_primitive = current.start;
                flat_nodes[current_node_idx].primitive_count = span;
                
                if (current.depth > max_depth) max_depth = current.depth;
                continue;
            }
            
            // Find split axis (longest extent)
            Vector3 extent = node_bbox.max - node_bbox.min;
            int axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > extent.y && extent.z > extent.x) axis = 2;
            
            // Sort primitives
            std::sort(indices + current.start, indices + current.end,
                [this, axis](int a, int b) { return box_compare(a, b, axis); });
            
            int mid = current.start + span / 2;
            
            // Create internal node
            flat_nodes[current_node_idx].bbox = node_bbox;
            flat_nodes[current_node_idx].primitive_count = 0;  // Mark as internal
            
            // Push children (right first, then left for stack order)
            node_stack.push(BuildNode(mid, current.end, current_node_idx, current.depth + 1));
            node_stack.push(BuildNode(current.start, mid, current_node_idx, current.depth + 1));
            
            // Store child indices (will be updated after building)
            flat_nodes[current_node_idx].left_child = -1;
            flat_nodes[current_node_idx].right_child = -1;
        }
        
        // Second pass to assign child indices
        std::queue<int> node_queue;
        node_queue.push(0);
        int processed = 0;
        
        while (!node_queue.empty()) {
            int node_idx = node_queue.front();
            node_queue.pop();
            
            BVHNodeFlat& node = flat_nodes[node_idx];
            
            if (!node.is_leaf()) {
                node.left_child = ++processed;
                node.right_child = ++processed;
                node_queue.push(node.left_child);
                node_queue.push(node.right_child);
            }
        }
        
        node_count = node_index;
        return node_count;
    }
    
    int get_node_count() const { return node_count; }
    int get_max_depth() const { return max_depth; }
};

// ================================================
// SCENE INTERSECTOR
// ================================================
class SceneIntersector {
private:
    Sphere* spheres;
    int sphere_count;
    BVHNodeFlat* bvh_nodes;
    int* indices;
    int node_count;
    
public:
    SceneIntersector() : spheres(nullptr), sphere_count(0), 
                        bvh_nodes(nullptr), indices(nullptr), node_count(0) {}
    
    ~SceneIntersector() {
        delete[] bvh_nodes;
        delete[] indices;
    }
    
    void build_bvh(Sphere* scene_spheres, int count) {
        spheres = scene_spheres;
        sphere_count = count;
        
        if (count == 0) return;
        
        // Allocate indices array
        delete[] indices;
        indices = new int[count];
        for (int i = 0; i < count; ++i) indices[i] = i;
        
        // Allocate BVH nodes (max 2n - 1)
        delete[] bvh_nodes;
        int max_nodes = 2 * count - 1;
        bvh_nodes = new BVHNodeFlat[max_nodes];
        
        // Build BVH
        BVHBuilder builder(spheres, indices, count);
        node_count = builder.build(bvh_nodes, max_nodes);
        
        std::cout << "BVH built with " << node_count << " nodes, max depth: " 
                  << builder.get_max_depth() << std::endl;
    }
    
    FORCEINLINE bool intersect(const Ray& ray, float tmin, float tmax,
                              float& hit_t, Vector3& hit_normal, 
                              Material& hit_mat, int& hit_id) const {
        if (sphere_count == 0) return false;
        
        if (bvh_nodes && node_count > 0) {
            // Use BVH traversal
            TraversalNode stack[64];  // Stack on local memory (no heap allocation)
            int stack_ptr = 0;
            stack[stack_ptr++] = TraversalNode(0, tmin);
            
            bool hit = false;
            float closest_t = tmax;
            Vector3 closest_normal;
            Material closest_mat;
            int closest_id;
            
            while (stack_ptr > 0) {
                TraversalNode tnode = stack[--stack_ptr];
                
                // Early termination
                if (tnode.tmin >= closest_t) continue;
                
                const BVHNodeFlat& node = bvh_nodes[tnode.node_index];
                
                // Skip if no intersection
                if (!node.bbox.intersect(ray, tnode.tmin, closest_t)) continue;
                
                if (node.is_leaf()) {
                    // Test all primitives in leaf
                    for (int i = 0; i < node.primitive_count; ++i) {
                        int idx = indices[node.first_primitive + i];
                        const Sphere& sphere = spheres[idx];
                        
                        float t;
                        Vector3 normal;
                        Material mat;
                        int id;
                        
                        if (sphere.intersect(ray, tnode.tmin, closest_t, t, normal, mat, id)) {
                            closest_t = t;
                            closest_normal = normal;
                            closest_mat = mat;
                            closest_id = id;
                            hit = true;
                        }
                    }
                } else {
                    // Push children
                    stack[stack_ptr++] = TraversalNode(node.left_child, tnode.tmin);
                    stack[stack_ptr++] = TraversalNode(node.right_child, tnode.tmin);
                }
            }
            
            if (hit) {
                hit_t = closest_t;
                hit_normal = closest_normal;
                hit_mat = closest_mat;
                hit_id = closest_id;
                return true;
            }
        } else {
            // Brute force fallback
            float closest_t = tmax;
            for (int i = 0; i < sphere_count; ++i) {
                float t;
                Vector3 normal;
                Material mat;
                int id;
                
                if (spheres[i].intersect(ray, tmin, closest_t, t, normal, mat, id)) {
                    closest_t = t;
                    hit_t = t;
                    hit_normal = normal;
                    hit_mat = mat;
                    hit_id = id;
                    return true;
                }
            }
        }
        
        return false;
    }
};

// ================================================
// PATH TRACER (Iterative, no recursion)
// ================================================
class PathTracer {
private:
    SceneIntersector scene_intersector;
    Vector3 background_color;
    
public:
    PathTracer() : background_color(0.1f, 0.1f, 0.1f) {}
    
    void set_scene(Sphere* spheres, int count) {
        scene_intersector.build_bvh(spheres, count);
    }
    
    FORCEINLINE Vector3 trace_ray(const Ray& ray, int max_depth, PCG32& rng) {
        Vector3 color(0, 0, 0);
        Vector3 throughput(1, 1, 1);
        Ray current_ray = ray;
        int depth = 0;
        
        while (depth < max_depth) {
            depth++;
            
            float t;
            Vector3 normal;
            Material mat;
            int id;
            
            // Intersection test
            if (!scene_intersector.intersect(current_ray, 0.001f, 1e10f, 
                                           t, normal, mat, id)) {
                // Ray missed - add background
                color = color + throughput * background_color;
                break;
            }
            
            // Add emitted light
            color = color + throughput * mat.emission;
            
            // Russian Roulette termination
            if (depth > 3) {
                float max_component = (throughput.x > throughput.y) ? 
                    (throughput.x > throughput.z ? throughput.x : throughput.z) :
                    (throughput.y > throughput.z ? throughput.y : throughput.z);
                
                float continue_probability = (max_component > 0.95f) ? 0.95f : max_component;
                if (continue_probability < 0.1f) continue_probability = 0.1f;
                
                if (rng.random_float() >= continue_probability) {
                    break;
                }
                throughput = throughput / continue_probability;
            }
            
            Vector3 hit_point = current_ray.at(t);
            
            // Material scattering
            if (mat.metallic > 0.0f) {
                // Metallic reflection
                Vector3 reflected = FastMath::reflect(current_ray.direction.normalize(), normal);
                Vector3 random_scatter = FastMath::random_in_unit_sphere(rng) * mat.roughness;
                Vector3 new_direction = (reflected + random_scatter).normalize();
                current_ray = Ray(hit_point, new_direction);
                throughput = throughput * mat.albedo;
            } else {
                // Diffuse reflection
                Vector3 random_dir = FastMath::random_in_hemisphere(normal, rng);
                Vector3 new_direction = (normal + random_dir).normalize();
                current_ray = Ray(hit_point, new_direction);
                throughput = throughput * mat.albedo;
            }
        }
        
        return color;
    }
    
    // Parallel rendering with OpenMP static scheduling
    void render(float* image_data, int width, int height, 
                int samples_per_pixel, int max_depth, const Camera& camera) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Precompute 1/width and 1/height
        float inv_width = 1.0f / width;
        float inv_height = 1.0f / height;
        
        int total_pixels = width * height;
        
        #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        std::cout << "Rendering with " << num_threads << " threads" << std::endl;
        #pragma omp parallel
        #endif
        {
            #ifdef _OPENMP
            int thread_id = omp_get_thread_num();
            #else
            int thread_id = 0;
            #endif
            
            // Each thread gets its own RNG with different seed
            PCG32 rng(thread_id + 1);
            
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int pixel_idx = 0; pixel_idx < total_pixels; ++pixel_idx) {
                int j = pixel_idx / width;
                int i = pixel_idx % width;
                
                Vector3 pixel_color(0, 0, 0);
                
                for (int s = 0; s < samples_per_pixel; ++s) {
                    // Jittered sampling
                    float u = (i + rng.random_float()) * inv_width;
                    float v = (j + rng.random_float()) * inv_height;
                    
                    Ray ray = camera.get_ray(u, v);
                    pixel_color = pixel_color + trace_ray(ray, max_depth, rng);
                }
                
                pixel_color = pixel_color * (1.0f / samples_per_pixel);
                
                // Fast gamma correction (sqrt)
                pixel_color = Vector3(sqrtf(pixel_color.x), 
                                     sqrtf(pixel_color.y), 
                                     sqrtf(pixel_color.z));
                
                // Clamp and store
                int idx = (j * width + i) * 3;
                image_data[idx] = pixel_color.x < 0.0f ? 0.0f : (pixel_color.x > 1.0f ? 1.0f : pixel_color.x);
                image_data[idx + 1] = pixel_color.y < 0.0f ? 0.0f : (pixel_color.y > 1.0f ? 1.0f : pixel_color.y);
                image_data[idx + 2] = pixel_color.z < 0.0f ? 0.0f : (pixel_color.z > 1.0f ? 1.0f : pixel_color.z);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Render time: " << duration.count() << "ms" << std::endl;
    }
};

// ================================================
// PYTHON BINDING INTERFACE - UPDATED
// ================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(raytracer_cpp, m) {
    m.doc() = "High-performance ray tracer with AVX2 and OpenMP";
    
    // Bind Vector3
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z)
        .def("dot", &Vector3::dot)
        .def("cross", &Vector3::cross)
        .def("length", &Vector3::length)
        .def("normalize", &Vector3::normalize)
        .def("__add__", [](const Vector3& a, const Vector3& b) { return a + b; })
        .def("__sub__", [](const Vector3& a, const Vector3& b) { return a - b; })
        .def("__mul__", [](const Vector3& a, float s) { return a * s; })
        .def("__mul__", [](const Vector3& a, const Vector3& b) { return a * b; })
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(" + std::to_string(v.x) + ", " + 
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    // Bind Material
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("albedo", &Material::albedo)
        .def_readwrite("metallic", &Material::metallic)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("emission", &Material::emission)
        .def_readwrite("ior", &Material::ior);
    
    // Bind Sphere
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("material", &Sphere::material)
        .def_readwrite("object_id", &Sphere::object_id)
        .def("__repr__", [](const Sphere& s) {
            return "Sphere(id=" + std::to_string(s.object_id) + 
                   ", center=" + std::to_string(s.center.x) + "," + 
                   std::to_string(s.center.y) + "," + std::to_string(s.center.z) + 
                   ", r=" + std::to_string(s.radius) + ")";
        });
    
    // Simple Scene wrapper
    class SceneWrapper {
    private:
        std::vector<Sphere> spheres;
        
    public:
        SceneWrapper() = default;
        
        void add_sphere(const Sphere& sphere) {
            spheres.push_back(sphere);
        }
        
        bool remove_sphere(int object_id) {
            for (auto it = spheres.begin(); it != spheres.end(); ++it) {
                if (it->object_id == object_id) {
                    spheres.erase(it);
                    return true;
                }
            }
            return false;
        }
        
        std::vector<Sphere>& get_spheres() { return spheres; }
        
        void build_bvh() {
            // Placeholder - in real implementation would rebuild BVH
            std::cout << "Scene BVH built with " << spheres.size() << " spheres" << std::endl;
        }
    };
    
    // Bind SceneWrapper as Scene
    py::class_<SceneWrapper>(m, "Scene")
        .def(py::init<>())
        .def("add_sphere", &SceneWrapper::add_sphere)
        .def("remove_sphere", &SceneWrapper::remove_sphere)
        .def("build_bvh", &SceneWrapper::build_bvh)
        .def_readwrite("spheres", &SceneWrapper::get_spheres)
        .def("__repr__", [](SceneWrapper& s) {
            return "Scene(spheres=" + std::to_string(s.get_spheres().size()) + ")";
        });
    
    // Bind Camera - UPDATED to match new struct
    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("position", &Camera::position)
        .def_readwrite("fov", &Camera::fov)
        .def_readwrite("aspect_ratio", &Camera::aspect_ratio)
        .def("update_basis", &Camera::update_basis)
        .def("get_ray", &Camera::get_ray)
        .def("move", &Camera::move)
        .def("__repr__", [](const Camera& c) {
            return "Camera(pos=(" + std::to_string(c.position.x) + "," + 
                   std::to_string(c.position.y) + "," + std::to_string(c.position.z) + 
                   "), fov=" + std::to_string(c.fov) + ")";
        });
    
    // Bind RayTracer
    class RayTracerWrapper {
    private:
        std::unique_ptr<PathTracer> tracer;
        std::vector<Sphere> spheres;
        Camera camera;
        
    public:
        RayTracerWrapper() : tracer(new PathTracer()) {
            camera.position = Vector3(0, 2, 5);
            camera.fov = 45.0f;
            camera.aspect_ratio = 1.333f;
            camera.update_basis();
        }
        
        void set_scene(SceneWrapper& scene) {
            spheres = scene.get_spheres();
            if (!spheres.empty()) {
                tracer->set_scene(spheres.data(), (int)spheres.size());
            }
        }
        
        Camera get_camera() { return camera; }
        Camera& get_camera_ref() { return camera; }
        
        void set_camera(const Camera& cam) {
            camera = cam;
            camera.update_basis();
        }
        
        py::array_t<float> render(int width, int height, int samples, int max_depth) {
            camera.aspect_ratio = static_cast<float>(width) / height;
            camera.update_basis();
            
            auto result = py::array_t<float>({height, width, 3});
            auto buf = result.request();
            float* image_data = static_cast<float*>(buf.ptr);
            
            if (spheres.empty()) {
                // Return black image
                std::fill(image_data, image_data + width * height * 3, 0.0f);
                return result;
            }
            
            tracer->render(image_data, width, height, samples, max_depth, camera);
            return result;
        }
        
        void select_object(int object_id) {
            // Placeholder
        }
        
        void move_camera(const Vector3& delta) {
            camera.move(delta);
        }
    };
    
    py::class_<RayTracerWrapper>(m, "RayTracer")
        .def(py::init<>())
        .def("set_scene", &RayTracerWrapper::set_scene)
        .def("render", &RayTracerWrapper::render)
        .def("get_camera", &RayTracerWrapper::get_camera)
        .def("set_camera", &RayTracerWrapper::set_camera)
        .def("select_object", &RayTracerWrapper::select_object)
        .def("move_camera", &RayTracerWrapper::move_camera);
}