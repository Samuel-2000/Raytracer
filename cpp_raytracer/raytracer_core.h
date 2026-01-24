#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <memory>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE3
#include "textures.h"

// Add material types
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

// Optimized Vector3 with same interface
struct Vector3 {
    double x, y, z;
    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    double operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    double& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    // OPTIMIZED: Inline operators for speed
    Vector3 operator+(const Vector3& other) const { 
        return Vector3(x + other.x, y + other.y, z + other.z); 
    }
    
    Vector3 operator-(const Vector3& other) const { 
        return Vector3(x - other.x, y - other.y, z - other.z); 
    }
    
    Vector3 operator*(double scalar) const { 
        return Vector3(x * scalar, y * scalar, z * scalar); 
    }
    
    Vector3 operator*(const Vector3& other) const { 
        return Vector3(x * other.x, y * other.y, z * other.z); 
    }
    
    Vector3 operator/(double scalar) const { 
        double inv_scalar = 1.0 / scalar;  // One division instead of three
        return Vector3(x * inv_scalar, y * inv_scalar, z * inv_scalar); 
    }
    
    Vector3 operator-() const { 
        return Vector3(-x, -y, -z); 
    }
    
    Vector3& operator+=(const Vector3& other) { 
        x += other.x; y += other.y; z += other.z; 
        return *this; 
    }
    
    Vector3& operator*=(double scalar) { 
        x *= scalar; y *= scalar; z *= scalar; 
        return *this; 
    }

    // OPTIMIZED: Fast dot product
    double dot(const Vector3& other) const { 
        return x * other.x + y * other.y + z * other.z; 
    }
    
    Vector3 cross(const Vector3& other) const { 
        return Vector3(y * other.z - z * other.y, 
                      z * other.x - x * other.z, 
                      x * other.y - y * other.x);
    }
    
    double length_squared() const { 
        return x*x + y*y + z*z; 
    }
    
    double length() const { 
        return std::sqrt(length_squared()); 
    }
    
    Vector3 normalize() const { 
        double len = length(); 
        if (len > 0) {
            double inv_len = 1.0 / len;
            return Vector3(x * inv_len, y * inv_len, z * inv_len);
        }
        return *this;
    }
};

// Add these optimized helper functions
inline Vector3 operator*(double scalar, const Vector3& vec) {
    return vec * scalar;
}

inline Vector3 lerp(const Vector3& a, const Vector3& b, double t) {
    return a * (1.0 - t) + b * t;
}

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
    
    // Material type for presets
    MaterialType material_type;
    
    Material() : albedo(0.8, 0.8, 0.8), metallic(0.0), roughness(0.5), 
                emission(0,0,0), ior(1.5), material_type(MATERIAL_CUSTOM) {}
};

struct HitRecord {
    double t;
    Vector3 point;
    Vector3 normal;
    Vector3 sphere_center;  // ADDED for texture mapping
    Material material;
    bool front_face;
    int object_id;
    
    HitRecord() : t(0), point(0,0,0), normal(0,0,0), sphere_center(0,0,0), material(), 
                 front_face(true), object_id(0) {}
    
    void set_face_normal(const Ray& ray, const Vector3& outward_normal) {
        front_face = ray.direction.dot(outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1.0;
    }
};

struct Sphere {
    Vector3 center;
    double radius;
    Material material;
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
    
    Camera() : position(0, 1, 5), target(0, 1, 4), up(0, 1, 0), fov(45.0), aspect_ratio(16/9) {}
    
    Ray get_ray(double u, double v) const {
        // Convert from [0,1] to [-1,1] and account for aspect ratio
        double ndc_x = (u - 0.5) * 2.0;
        double ndc_y = (0.5 - v) * 2.0;  // Flip Y
        
        // Compute camera basis vectors based on target
        Vector3 forward = (target - position).normalize();
        Vector3 right = forward.cross(Vector3(0, 1, 0)).normalize();
        if (right.length() < 0.001) {
            right = Vector3(1, 0, 0);
        }
        Vector3 up = right.cross(forward).normalize();
        
        // Calculate the viewport dimensions based on FOV
        double fov_rad = fov * 0.00872664925997222; // 3.14159265359 / 360.0;  // Convert to radians
        double viewport_height = std::tan(fov_rad);
        double viewport_width = viewport_height * aspect_ratio;
        
        // Apply the viewport scaling to ndc coordinates
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
        // Simple rotation around target
        Vector3 forward = (target - position).normalize();
        Vector3 right = forward.cross(up).normalize();
        
        // Rotate position around target
        double distance = (target - position).length();
        Vector3 offset = position - target;
        
        // Apply rotation (simplified)
        position = target + offset;
    }
};

class BVH;
class Skybox;

struct DebugInfo {
    bool enable_debug = false;
    int build_count = 0;
    int render_count = 0;
    
    void reset() {
        build_count = 0;
        render_count = 0;
    }
    
    std::string get_stats() const {
        return "Builds: " + std::to_string(build_count) + 
               ", Renders: " + std::to_string(render_count);
    }
};

class Scene {
public:
    std::vector<Sphere> spheres;
    std::unique_ptr<Skybox> skybox;  // NEW
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
    
    // Skybox methods
    Skybox* get_skybox() { return skybox.get(); }
    void set_skybox(std::unique_ptr<Skybox> new_skybox) { skybox = std::move(new_skybox); }
};

class RayTracer {
private:
    Scene scene;
    Camera camera;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    DebugInfo debug_info;
    
    Vector3 random_in_unit_sphere();
    Vector3 random_in_hemisphere(const Vector3& normal);
    Vector3 reflect(const Vector3& v, const Vector3& n);
    bool refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted);
    double schlick(double cosine, double ref_idx);
    
    // Texture sampling
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

    void set_debug_mode(bool enable) { debug_info.enable_debug = enable; }
    DebugInfo get_debug_info() const { return debug_info; }
    
    // Scene access
    Scene& get_scene() { return scene; }
};