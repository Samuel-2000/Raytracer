// raytracer_core.h - UPDATED TO INCLUDE material.h
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <memory>
#include "vector3.h"
#include "material.h"  // ADD THIS LINE

// Forward declarations
class Texture;
class Skybox;
class BVH;

struct Ray {
    Vector3 origin;
    Vector3 direction;
    
    Ray() : origin(0,0,0), direction(0,0,0) {}
    Ray(const Vector3& orig, const Vector3& dir) : origin(orig), direction(dir.normalize()) {}
    
    Vector3 at(double t) const { 
        return origin + direction * t; 
    }
};

struct HitRecord {
    double t;
    Vector3 point;
    Vector3 normal;
    Vector3 sphere_center;
    std::shared_ptr<Texture> material_albedo_texture;
    std::shared_ptr<Texture> material_roughness_texture;
    Vector3 albedo;
    double metallic;
    double roughness;
    Vector3 emission;
    double ior;
    bool front_face;
    int object_id;
    
    HitRecord() : t(0), point(0,0,0), normal(0,0,0), sphere_center(0,0,0),
                 albedo(0.8,0.8,0.8), metallic(0.0), roughness(0.5),
                 emission(0,0,0), ior(1.5), front_face(true), object_id(0) {}
    
    void set_face_normal(const Ray& ray, const Vector3& outward_normal) {
        front_face = ray.direction.dot(outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1.0;
    }
};

struct Sphere {
    Vector3 center;
    double radius;
    Material material;  // CHANGED FROM INDIVIDUAL FIELDS TO Material STRUCT
    int object_id;
    std::string name;
    
    Sphere() : center(0,0,0), radius(1.0), material(), object_id(0), name("") {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
};

class Camera {
public:
    Vector3 position;
    Vector3 target;
    Vector3 up;
    double fov;
    double aspect_ratio;
    
    Camera() : position(0, 1, 5), target(0, 1, 4), up(0, 1, 0), fov(45.0), aspect_ratio(16.0/9.0) {}
    
    Ray get_ray(double u, double v) const;
        
    void move(const Vector3& delta) {
        position = position + delta;
    }
    
    void rotate(double dx, double dy) {
        // Simple rotation
        Vector3 forward = (target - position).normalize();
        Vector3 right = forward.cross(up).normalize();
        
        double distance = (target - position).length();
        Vector3 offset = position - target;
        
        // Apply rotation
        position = target + offset;
    }
};

class Scene {
public:
    std::vector<Sphere> spheres;
    Skybox* skybox;
    Vector3 background_color;
    bool use_bvh;
    bool debug_mode;
    BVH* bvh;
    
    Scene();
    Scene(const Scene& other);
    Scene& operator=(const Scene& other);
    ~Scene();
    
    void add_sphere(const Sphere& sphere);
    void remove_sphere(int object_id);
    void build_bvh();
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
    int cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const;

    void set_skybox(Skybox* new_skybox);
    Skybox* get_skybox() const;
};

class RayTracer {
private:
    Scene scene;
    Camera camera;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
    Vector3 random_in_unit_sphere();
    Vector3 random_in_hemisphere(const Vector3& normal);
    Vector3 reflect(const Vector3& v, const Vector3& n);
    bool refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted);
    double schlick(double cosine, double ref_idx);
    
    Vector3 sample_albedo(const HitRecord& rec) const;
    float sample_roughness(const HitRecord& rec) const;
    
public:
    RayTracer();
    ~RayTracer();
    void set_scene(const Scene& new_scene);
    Vector3 trace_ray(const Ray& ray, int depth, int max_depth);
    std::vector<double> render(int width, int height, int samples_per_pixel, int max_depth);

    Camera& get_camera() { return camera; }
    Camera get_camera_copy() const { return camera; }
    void set_camera(const Camera& cam) { camera = cam; }
    int select_object(double x, double y, int width, int height);
    void move_camera(const Vector3& delta);
    
    Scene& get_scene() { return scene; }
};