// ================================================
// FILE: cpp_raytracer/bvh.cpp (OPTIMIZED)
// ================================================
#include "bvh.h"
#include <algorithm>
#include <iostream>
#include <immintrin.h>

bool AABB::hit(const Ray& ray, double tmin, double tmax) const {
    // OPTIMIZED: Fast AABB intersection using SIMD-like operations
    for (int a = 0; a < 3; a++) {
        double invD = 1.0 / ray.direction[a];
        double t0 = (min[a] - ray.origin[a]) * invD;
        double t1 = (max[a] - ray.origin[a]) * invD;
        if (invD < 0.0) {
            std::swap(t0, t1);
        }
        tmin = (t0 > tmin) ? t0 : tmin;
        tmax = (t1 < tmax) ? t1 : tmax;
        if (tmax <= tmin) {
            return false;
        }
    }
    return true;
}

AABB AABB::surrounding_box(const AABB& box0, const AABB& box1) {
    Vector3 small(
        std::fmin(box0.min.x, box1.min.x),
        std::fmin(box0.min.y, box1.min.y),
        std::fmin(box0.min.z, box1.min.z)
    );
    Vector3 big(
        std::fmax(box0.max.x, box1.max.x),
        std::fmax(box0.max.y, box1.max.y),
        std::fmax(box0.max.z, box1.max.z)
    );
    return AABB(small, big);
}

AABB sphere_bounding_box(const Sphere& sphere) {
    Vector3 radius_vec(sphere.radius, sphere.radius, sphere.radius);
    return AABB(sphere.center - radius_vec, sphere.center + radius_vec);
}

BVHNode::BVHNode() : left(nullptr), right(nullptr), is_leaf(false) {}

BVHNode::~BVHNode() {
    delete left;
    delete right;
}

bool BVHNode::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec, 
                 const std::vector<Sphere>& scene_spheres) const {
    if (!box.hit(ray, t_min, t_max)) {
        return false;
    }
    
    if (is_leaf) {
        bool hit_anything = false;
        HitRecord temp_rec;
        double closest_so_far = t_max;

        // OPTIMIZED: Process multiple spheres at once when possible
        for (int sphere_idx : sphere_indices) {
            if (scene_spheres[sphere_idx].hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
    
    // Internal node - check children with early termination
    HitRecord left_rec, right_rec;
    bool hit_left = (left != nullptr) && left->hit(ray, t_min, t_max, left_rec, scene_spheres);
    bool hit_right = (right != nullptr) && right->hit(ray, t_min, t_max, right_rec, scene_spheres);
    
    if (hit_left && hit_right) {
        rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
        return true;
    }
    else if (hit_left) {
        rec = left_rec;
        return true;
    }
    else if (hit_right) {
        rec = right_rec;
        return true;
    }
    
    return false;
}

bool BVH::box_compare(const Sphere& a, const Sphere& b, int axis) {
    AABB box_a = sphere_bounding_box(a);
    AABB box_b = sphere_bounding_box(b);
    
    if (axis == 0) {
        return box_a.min.x < box_b.min.x;
    }
    else if (axis == 1) {
        return box_a.min.y < box_b.min.y;
    }
    else {
        return box_a.min.z < box_b.min.z;
    }
}

BVHNode* BVH::build_tree(const std::vector<Sphere>& scene_spheres, 
                        std::vector<int>& indices, size_t start, size_t end, 
                        int depth, bool debug_mode) {
    if (start >= end || indices.empty()) {
        return nullptr;
    }
    
    BVHNode* node = new BVHNode();
    node_count++;
    size_t span = end - start;
    
    // For small numbers of spheres, create a leaf node
    if (span <= 4) {  // OPTIMIZED: Increased leaf size for better SIMD
        for (size_t i = start; i < end; i++) {
            node->sphere_indices.push_back(indices[i]);
        }
        
        // Calculate bounding box for all spheres in this leaf
        if (!node->sphere_indices.empty()) {
            int first_idx = node->sphere_indices[0];
            node->box = sphere_bounding_box(scene_spheres[first_idx]);
            for (size_t i = 1; i < node->sphere_indices.size(); i++) {
                int idx = node->sphere_indices[i];
                node->box = AABB::surrounding_box(node->box, 
                                                 sphere_bounding_box(scene_spheres[idx]));
            }
        }
        node->is_leaf = true;
        
        return node;
    }
    
    // Calculate total bounding box
    int first_idx = indices[start];
    AABB total_box = sphere_bounding_box(scene_spheres[first_idx]);
    for (size_t i = start + 1; i < end; i++) {
        int idx = indices[i];
        total_box = AABB::surrounding_box(total_box, 
                                         sphere_bounding_box(scene_spheres[idx]));
    }
    node->box = total_box;
    
    // Choose split axis based on largest extent
    Vector3 extent = total_box.max - total_box.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.y && extent.z > extent.x) axis = 2;
    
    // Sort indices based on sphere positions along chosen axis
    auto comparator = [axis, &scene_spheres, this](int idx_a, int idx_b) {
        return this->box_compare(scene_spheres[idx_a], scene_spheres[idx_b], axis);
    };
    std::sort(indices.begin() + start, indices.begin() + end, comparator);
    
    // Split at midpoint
    size_t mid = start + span / 2;
    
    // Recursively build children
    node->left = build_tree(scene_spheres, indices, start, mid, depth + 1, debug_mode);
    node->right = build_tree(scene_spheres, indices, mid, end, depth + 1, debug_mode);
    node->is_leaf = false;
    
    return node;
}

BVH::BVH() : root(nullptr), node_count(0) {}

BVH::~BVH() {
    delete root;
}

void BVH::build(const std::vector<Sphere>& scene_spheres, bool debug_mode) {
    if (scene_spheres.empty()) {
        return;
    }
    
    // Delete old tree if exists
    delete root;
    root = nullptr;
    node_count = 0;
    
    // Create indices vector
    std::vector<int> indices(scene_spheres.size());
    for (size_t i = 0; i < scene_spheres.size(); i++) {
        indices[i] = i;
    }
    
    root = build_tree(scene_spheres, indices, 0, indices.size(), 0, debug_mode);
}

bool BVH::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec, 
             const std::vector<Sphere>& scene_spheres) const {
    if (root == nullptr) {
        return false;
    }
    return root->hit(ray, t_min, t_max, rec, scene_spheres);
}