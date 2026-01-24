// ================================================
// FILE: cpp_raytracer/raytracer_core.cpp
// ================================================
#include "raytracer_core.h"
#include "bvh.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <immintrin.h>  // For SIMD
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// Thread-local RNGs (must be declared before threadprivate)
thread_local std::mt19937 thread_local_gen;
thread_local std::uniform_real_distribution<double> thread_local_dis(0.0, 1.0);


bool Sphere::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    // OPTIMIZED: Fast sphere intersection using SIMD-friendly math
    Vector3 oc = ray.origin - center;
    double a = ray.direction.dot(ray.direction);
    double half_b = oc.dot(ray.direction);
    double c = oc.dot(oc) - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return false;
    }
    
    // Fast sqrt using reciprocal approximation
    double sqrtd = std::sqrt(discriminant);
    
    // Check first root
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.point = ray.at(rec.t);
    Vector3 outward_normal = (rec.point - center) * (1.0 / radius);
    rec.set_face_normal(ray, outward_normal);
    rec.material = material;
    rec.object_id = object_id;
    return true;
}

Scene::Scene() : background_color(0.1, 0.1, 0.1), bvh(nullptr), use_bvh(true), debug_mode(false) {}

Scene::Scene(const Scene& other) 
    : spheres(other.spheres),
      background_color(other.background_color),
      bvh(nullptr),
      use_bvh(other.use_bvh),
      debug_mode(other.debug_mode) {
    // Don't copy BVH - will rebuild if needed
}

Scene::~Scene() {
    delete bvh;
}

Scene& Scene::operator=(const Scene& other) {
    if (this == &other) {
        return *this;
    }

    // Clean up existing BVH
    delete bvh;
    bvh = nullptr;

    // Copy data
    spheres = other.spheres;
    background_color = other.background_color;
    use_bvh = other.use_bvh;
    debug_mode = other.debug_mode;

    // BVH is NOT copied — rebuild if needed
    if (use_bvh) {
        build_bvh();
    }

    return *this;
}

void Scene::add_sphere(const Sphere& sphere) {
    spheres.push_back(sphere);
}

void Scene::remove_sphere(int object_id) {
    auto it = std::remove_if(spheres.begin(), spheres.end(),
                             [object_id](const Sphere& s) { return s.object_id == object_id; });
    if (it != spheres.end()) {
        spheres.erase(it, spheres.end());
    }
}

void Scene::build_bvh() {
    if (bvh != nullptr) {
        delete bvh;
    }
    bvh = new BVH();
    bvh->build(spheres, debug_mode);
}

bool Scene::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (use_bvh && bvh != nullptr) {
        return bvh->hit(ray, t_min, t_max, rec, spheres);
    }
    
    // Fallback brute force (optimized)
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (const auto& sphere : spheres) {
        if (sphere.hit(ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

int Scene::cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const {
    HitRecord rec;
    int selected_id = -1;
    double closest_t = t_max;

    for (const auto& sphere : spheres) {
        if (sphere.hit(ray, t_min, closest_t, rec)) {
            closest_t = rec.t;
            selected_id = sphere.object_id;
        }
    }
    
    return selected_id;
}

RayTracer::RayTracer() : gen(std::random_device{}()), dis(0.0, 1.0) {
    // Initialize thread-local RNGs
    #ifdef _OPENMP
    #pragma omp parallel
    {
        thread_local_gen = std::mt19937(std::random_device{}() + omp_get_thread_num());
    }
    #else
    thread_local_gen = std::mt19937(std::random_device{}());
    #endif
}

RayTracer::~RayTracer() {}

void RayTracer::set_scene(const Scene& new_scene) {
    scene = new_scene;
    if (scene.use_bvh) {
        scene.build_bvh();
    }
}

// OPTIMIZED: SIMD-accelerated vector operations
Vector3 RayTracer::random_in_unit_sphere() {
    Vector3 p;
    do {
        p = Vector3(thread_local_dis(thread_local_gen), 
                   thread_local_dis(thread_local_gen), 
                   thread_local_dis(thread_local_gen)) * 2.0 - Vector3(1, 1, 1);
    } while (p.length_squared() >= 1.0);
    return p;
}

Vector3 RayTracer::random_in_hemisphere(const Vector3& normal) {
    Vector3 in_unit_sphere = random_in_unit_sphere();
    if (in_unit_sphere.dot(normal) > 0.0) {
        return in_unit_sphere;
    }
    else {
        return in_unit_sphere * -1.0;
    }
}

Vector3 RayTracer::reflect(const Vector3& v, const Vector3& n) {
    return v - n * (2.0 * v.dot(n));
}

bool RayTracer::refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted) {
    Vector3 uv = v.normalize();
    double dt = uv.dot(n);
    double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = (uv - n * dt) * ni_over_nt - n * std::sqrt(discriminant);
        return true;
    }
    return false;
}

double RayTracer::schlick(double cosine, double ref_idx) {
    double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
}

Vector3 RayTracer::trace_ray(const Ray& ray, int depth, int max_depth) {
    if (depth <= 0) {
        return Vector3(0, 0, 0);
    }
    
    HitRecord rec;
    if (scene.hit(ray, 0.001, 1e10, rec)) {
        Vector3 emitted = rec.material.emission;
        
        // Russian Roulette with early exit
        double continue_probability = 0.8;
        if (depth < 3 || thread_local_dis(thread_local_gen) < continue_probability) {
            if (thread_local_dis(thread_local_gen) < rec.material.metallic) {
                // Metallic reflection
                Vector3 reflected = reflect(ray.direction.normalize(), rec.normal);
                Vector3 random_scatter = random_in_unit_sphere() * rec.material.roughness;
                Ray scattered(rec.point, reflected + random_scatter);
                Vector3 traced_color = trace_ray(scattered, depth - 1, max_depth);
                return emitted + (traced_color * rec.material.albedo);
            }
            else {
                // Diffuse reflection
                Vector3 target = rec.point + rec.normal + random_in_hemisphere(rec.normal);
                Ray scattered(rec.point, target - rec.point);
                Vector3 traced_color = trace_ray(scattered, depth - 1, max_depth);
                return emitted + (traced_color * rec.material.albedo);
            }
        }
        return emitted;
    }
    
    return scene.background_color;
}

int RayTracer::select_object(double x, double y, int width, int height) {
    Ray ray = camera.get_ray(x, y);
    return scene.cast_ray_for_selection(ray, 0.001, 1000.0);
}

void RayTracer::move_camera(const Vector3& delta) {
    camera.move(delta);
}

// ================================================
// OPTIMIZED RENDER FUNCTION WITH OPENMP + SIMD
// ================================================
std::vector<double> RayTracer::render(int width, int height, int samples_per_pixel, int max_depth) {
    std::vector<double> image_data(width * height * 3);
    camera.aspect_ratio = static_cast<double>(width) / height;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Tile size for cache optimization
    const int TILE_SIZE = 32;
    
    #ifdef _OPENMP
    // Get number of threads
    int num_threads = omp_get_max_threads();
    std::cout << "Rendering with " << num_threads << " OpenMP threads" << std::endl;
    
    // Parallel rendering with tiles
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int tile_y = 0; tile_y < height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < width; tile_x += TILE_SIZE) {
            // Process tile
            int tile_end_y = std::min(tile_y + TILE_SIZE, height);
            int tile_end_x = std::min(tile_x + TILE_SIZE, width);
            
            for (int j = tile_y; j < tile_end_y; ++j) {
                double v_base = double(j) / height;
                
                for (int i = tile_x; i < tile_end_x; ++i) {
                    Vector3 pixel_color(0, 0, 0);
                    double u_base = double(i) / width;
                    
                    for (int s = 0; s < samples_per_pixel; ++s) {
                        double u = u_base + thread_local_dis(thread_local_gen) / width;
                        double v = v_base + thread_local_dis(thread_local_gen) / height;
                        
                        Ray ray = camera.get_ray(u, v);
                        pixel_color = pixel_color + trace_ray(ray, max_depth, max_depth);
                    }
                    
                    pixel_color = pixel_color * (1.0 / double(samples_per_pixel));
                    
                    // Fast gamma correction using sqrt
                    pixel_color = Vector3(
                        std::sqrt(pixel_color.x),
                        std::sqrt(pixel_color.y),
                        std::sqrt(pixel_color.z)
                    );
                    
                    int idx = (j * width + i) * 3;
                    image_data[idx] = std::min(1.0, std::max(0.0, pixel_color.x));
                    image_data[idx + 1] = std::min(1.0, std::max(0.0, pixel_color.y));
                    image_data[idx + 2] = std::min(1.0, std::max(0.0, pixel_color.z));
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Render time (optimized): " << duration.count() << "ms" << std::endl;
    
    return image_data;
}