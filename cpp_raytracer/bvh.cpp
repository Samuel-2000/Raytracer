#include "raytracer_core.h"
#include <algorithm>
#include <stack>
#include <queue>

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
// BVH TRAVERSAL (Stack-based, no heap allocations)
// ================================================
class BVHTraversal {
private:
    struct TraversalNode {
        int node_index;
        float tmin;
        
        FORCEINLINE TraversalNode(int idx, float t) : node_index(idx), tmin(t) {}
    };
    
public:
    // Iterative BVH traversal with early termination
    template<typename HitCallback>
    FORCEINLINE static bool intersect(const Ray& ray, float tmin, float tmax,
                                     const BVHNodeFlat* nodes, const Sphere* spheres,
                                     const int* indices, HitCallback&& hit_callback) {
        TraversalNode stack[64];  // Stack on local memory (no heap allocation)
        int stack_ptr = 0;
        stack[stack_ptr++] = TraversalNode(0, tmin);
        
        bool hit = false;
        float closest_t = tmax;
        
        while (stack_ptr > 0) {
            TraversalNode tnode = stack[--stack_ptr];
            
            // Early termination
            if (tnode.tmin >= closest_t) continue;
            
            const BVHNodeFlat& node = nodes[tnode.node_index];
            
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
                        hit = hit_callback(t, normal, mat, id);
                        if (hit) closest_t = t;
                    }
                }
            } else {
                // Push children in order based on ray direction
                int left_idx = node.left_child;
                int right_idx = node.right_child;
                
                // Optional: sort children by distance to ray origin
                stack[stack_ptr++] = TraversalNode(left_idx, tnode.tmin);
                stack[stack_ptr++] = TraversalNode(right_idx, tnode.tmin);
            }
        }
        
        return hit;
    }
};

// ================================================
// OPTIMIZED SCENE INTERSECTION
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
        
        if (bvh_nodes) {
            // Use BVH traversal
            bool hit = false;
            float closest_t = tmax;
            Vector3 closest_normal;
            Material closest_mat;
            int closest_id;
            
            auto hit_callback = [&](float t, const Vector3& normal, 
                                   const Material& mat, int id) -> bool {
                if (t < closest_t) {
                    closest_t = t;
                    closest_normal = normal;
                    closest_mat = mat;
                    closest_id = id;
                    return true;
                }
                return false;
            };
            
            if (BVHTraversal::intersect(ray, tmin, tmax, bvh_nodes, spheres, 
                                       indices, hit_callback)) {
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
    
    ~SceneIntersector() {
        delete[] bvh_nodes;
        delete[] indices;
    }
};