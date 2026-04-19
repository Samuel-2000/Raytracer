// bvh.h
#pragma once
#include "vector3.h"
#include <vector>
#include <functional>

struct Ray;
struct RayPacket;
struct Sphere;
struct HitRecord;

class AABB {
public:
    Vector3 min;
    Vector3 max;
    AABB() : min(0,0,0), max(0,0,0) {}
    AABB(const Vector3& a, const Vector3& b) : min(a), max(b) {}
    Vector3 center() const { return (min + max) * 0.5; }
    bool hit(const Ray& ray, double tmin, double tmax) const;
    int hit_packet(const RayPacket& packet, double tmin, double tmax_arr[4], int active_mask) const;
    static AABB surrounding_box(const AABB& box0, const AABB& box1);
};

AABB sphere_bounding_box(const Sphere& sphere);

class BVHNode {
public:
    AABB box;
    BVHNode* left;
    BVHNode* right;
    std::vector<int> sphere_indices;
    bool is_leaf;
    
    BVHNode();
    ~BVHNode();
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
             const std::vector<Sphere>& scene_spheres, bool use_simd) const;
    int hit_packet(const RayPacket& packet, double t_min, double t_max_arr[4],
                   HitRecord rec[4], const std::vector<Sphere>& scene_spheres,
                   bool use_simd, int active_mask = 0xF) const;
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
    void refit(const std::vector<Sphere>& scene_spheres);
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
             const std::vector<Sphere>& scene_spheres, bool use_simd) const;
    int hit_packet(const RayPacket& packet, double t_min, double t_max_arr[4],
                   HitRecord rec[4], const std::vector<Sphere>& scene_spheres,
                   bool use_simd, int active_mask = 0xF) const;
    int get_node_count() const { return node_count; }
};