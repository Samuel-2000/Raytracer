// bvh.h
#pragma once
#include "vector3.h"
#include <vector>

// Complete forward declarations
struct Ray;
struct Sphere;
struct HitRecord;

struct Ray {
    Vector3 origin;
    Vector3 direction;
    Ray(const Vector3& orig, const Vector3& dir) : origin(orig), direction(dir) {}
    Vector3 at(double t) const { 
        return origin + direction * t; 
    }
};

struct HitRecord {
    double t;
    Vector3 point;
    Vector3 normal;
    bool front_face;
    int object_id;
    
    HitRecord() : t(0), point(0,0,0), normal(0,0,0), front_face(true), object_id(0) {}
    
    void set_face_normal(const Ray& ray, const Vector3& outward_normal) {
        front_face = ray.direction.dot(outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1.0;
    }
};

struct Sphere {
    Vector3 center;
    double radius;
    int object_id;
    
    Sphere() : center(0,0,0), radius(1.0), object_id(0) {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
};

class AABB {
public:
    Vector3 min;
    Vector3 max;
    
    AABB() : min(Vector3(0,0,0)), max(Vector3(0,0,0)) {}
    AABB(const Vector3& a, const Vector3& b) : min(a), max(b) {}
    
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
    std::vector<int> sphere_indices;
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