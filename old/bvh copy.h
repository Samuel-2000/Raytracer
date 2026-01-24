// bvh.h - UPDATED
#pragma once
#include "raytracer_core.h"
#include <algorithm>
#include <vector>

class AABB {
public:
    Vector3 min;
    Vector3 max;
    
    AABB() : min(Vector3(0,0,0)), max(Vector3(0,0,0)) {}
    AABB(const Vector3& a, const Vector3& b) : min(a), max(b) {}
    
    // ADD THIS METHOD:
    Vector3 center() const {
        return (min + max) * 0.5;
    }
    
    bool hit(const Ray& ray, double tmin, double tmax) const;
    static AABB surrounding_box(const AABB& box0, const AABB& box1);
};

AABB sphere_bounding_box(const Sphere& sphere);

class BVHNode {
public:
    AABB box;
    BVHNode* left;
    BVHNode* right;
    std::vector<int> sphere_indices;  // Store indices into scene's sphere array
    bool is_leaf;
    
    BVHNode();
    ~BVHNode();
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
            const std::vector<Sphere>& scene_spheres) const;
};

class BVH {
private:
    BVHNode* root;
    int node_count;
    
    BVHNode* build_tree(const std::vector<Sphere>& scene_spheres,
                       std::vector<int>& indices, size_t start, size_t end, 
                       int depth, bool debug_mode);
    bool box_compare(const Sphere& a, const Sphere& b, int axis);
    
public:
    BVH();
    ~BVH();
    
    void build(const std::vector<Sphere>& scene_spheres, bool debug_mode = false);
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
            const std::vector<Sphere>& scene_spheres) const;
    
    int get_node_count() const { return node_count; }
};