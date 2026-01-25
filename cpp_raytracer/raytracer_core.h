// raytracer_core.h
#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <memory>
#include "vector3.h"  // Include our Vector3 header

// Forward declarations
class Texture;
class Skybox;

// Material types
enum MaterialType {
    MATERIAL_CUSTOM = 0,
    MATERIAL_DIFFUSE = 1,
    MATERIAL_METAL = 2,
    MATERIAL_DIELECTRIC = 3,
    MATERIAL_PLASTIC = 4,
    MATERIAL_WOOD = 5,
    MATERIAL_MARBLE = 6,
    MATERIAL_RUSTY_METAL = 7,
    MATERIAL_GLASS = 8,
    MATERIAL_MIRROR = 9,
    MATERIAL_RUBBER = 10
};

struct Ray {
    Vector3 origin;
    Vector3 direction;
    Ray(const Vector3& orig, const Vector3& dir) : origin(orig), direction(dir.normalize()) {}
    Vector3 at(double t) const { 
        return origin + direction * t; 
    }
};

struct Material {
    Vector3 albedo;
    double metallic;
    double roughness;
    Vector3 emission;
    double ior;
    
    // Texture support
    std::shared_ptr<Texture> albedo_texture;
    std::shared_ptr<Texture> roughness_texture;
    
    MaterialType material_type;
    
    Material() : albedo(0.8, 0.8, 0.8), metallic(0.0), roughness(0.5), 
                emission(0,0,0), ior(1.5), material_type(MATERIAL_CUSTOM) {}
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
    Vector3 albedo;
    double metallic;
    double roughness;
    Vector3 emission;
    double ior;
    std::shared_ptr<Texture> albedo_texture;
    std::shared_ptr<Texture> roughness_texture;
    MaterialType material_type;
    int object_id;
    std::string name;
    
    Sphere() : center(0,0,0), radius(1.0), albedo(0.8,0.8,0.8), 
               metallic(0.0), roughness(0.5), emission(0,0,0), ior(1.5),
               material_type(MATERIAL_CUSTOM), object_id(0), name("") {}
    
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
    
    Ray get_ray(double u, double v) const {
        // Convert from [0,1] to [-1,1] and account for aspect ratio
        double ndc_x = (u - 0.5) * 2.0;
        double ndc_y = (0.5 - v) * 2.0;  // Flip Y
        
        // Compute camera basis vectors
        Vector3 forward = (target - position).normalize();
        Vector3 right = forward.cross(Vector3(0, 1, 0)).normalize();
        if (right.length() < 0.001) {
            right = Vector3(1, 0, 0);
        }
        Vector3 up = right.cross(forward).normalize();
        
        // Calculate viewport dimensions
        double fov_rad = fov * 0.0174533;  // Convert to radians (3.14159/180.0)
        double viewport_height = std::tan(fov_rad / 2.0);
        double viewport_width = viewport_height * aspect_ratio;
        
        // Calculate ray direction
        Vector3 direction = forward + 
                           (right * (ndc_x * viewport_width)) + 
                           (up * (ndc_y * viewport_height));
        
        direction = direction.normalize();
        
        return Ray(position, direction);
    }
        
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

class BVH;
class Skybox;

class Scene {
public:
    std::vector<Sphere> spheres;
    std::shared_ptr<Skybox> skybox;  // Changed to shared_ptr
    Vector3 background_color;
    bool use_bvh;
    bool debug_mode;
    BVH* bvh;
    
    Scene();  // Default constructor
    Scene(const Scene& other);  // Copy constructor
    Scene& operator=(const Scene& other);  // Assignment operator
    ~Scene();
    
    void add_sphere(const Sphere& sphere);
    void remove_sphere(int object_id);
    void build_bvh();
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
    int cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const;
    
    Skybox* get_skybox() { return skybox.get(); }
    void set_skybox(std::shared_ptr<Skybox> new_skybox) { skybox = new_skybox; }
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