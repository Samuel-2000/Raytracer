#pragma once

#include <cmath>
#include <cstdint>
#include <cfloat>  // Added for FLT_MAX
#include <immintrin.h>  // For AVX2

// ================================================
// FAST RANDOM NUMBER GENERATOR (PCG32)
// ================================================
class PCG32 {
private:
    uint64_t state;
    uint64_t inc;
    
    static constexpr uint64_t multiplier = 6364136223846793005ULL;
    static constexpr uint64_t increment = 1442695040888963407ULL;
    
public:
    __forceinline PCG32(uint64_t seed = 0) {
        state = 0U;
        inc = (seed << 1u) | 1u;
        random();
        state += 0x853c49e6748fea9bULL;
        random();
    }
    
    __forceinline uint32_t random() {
        uint64_t oldstate = state;
        state = oldstate * multiplier + inc;
        uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = uint32_t(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-(int)rot) & 31));  // Fixed cast
    }
    
    __forceinline float random_float() {
        uint32_t r = random();
        return float(r) * 2.3283064365386963e-10f;  // 1.0 / 2^32
    }
    
    __forceinline float random_float(float min, float max) {
        return min + (max - min) * random_float();
    }
};

// Thread-local RNG
#if defined(_MSC_VER)
    #define THREAD_LOCAL __declspec(thread)
#else
    #define THREAD_LOCAL thread_local
#endif

// ================================================
// SIMD-FRIENDLY STRUCTURES (AVX2 alignment)
// ================================================
#ifdef _MSC_VER
#define ALIGN32 __declspec(align(32))
#define FORCEINLINE __forceinline
#else
#define ALIGN32 alignas(32)
#define FORCEINLINE __attribute__((always_inline)) inline
#endif

struct ALIGN32 Vector3 {
    float x, y, z;
    
    FORCEINLINE Vector3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    FORCEINLINE Vector3 operator+(const Vector3& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    FORCEINLINE Vector3 operator-(const Vector3& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    FORCEINLINE Vector3 operator*(float s) const { return Vector3(x * s, y * s, z * s); }
    FORCEINLINE Vector3 operator*(const Vector3& v) const { return Vector3(x * v.x, y * v.y, z * v.z); }
    FORCEINLINE Vector3 operator/(float s) const { float inv = 1.0f / s; return Vector3(x * inv, y * inv, z * inv); }
    
    FORCEINLINE Vector3& operator+=(const Vector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    FORCEINLINE Vector3& operator-=(const Vector3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    FORCEINLINE Vector3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    
    FORCEINLINE float dot(const Vector3& v) const { return x * v.x + y * v.y + z * v.z; }
    FORCEINLINE Vector3 cross(const Vector3& v) const {
        return Vector3(y * v.z - z * v.y,
                      z * v.x - x * v.z,
                      x * v.y - y * v.x);
    }
    
    FORCEINLINE float length_squared() const { return x*x + y*y + z*z; }
    FORCEINLINE float length() const { return sqrtf(length_squared()); }
    FORCEINLINE Vector3 normalize() const { 
        float len = length(); 
        return len > 0.0f ? *this * (1.0f / len) : Vector3(0.0f, 0.0f, 1.0f); 
    }
    
    FORCEINLINE Vector3 min(const Vector3& v) const { 
        return Vector3(x < v.x ? x : v.x, 
                      y < v.y ? y : v.y, 
                      z < v.z ? z : v.z); 
    }
    FORCEINLINE Vector3 max(const Vector3& v) const { 
        return Vector3(x > v.x ? x : v.x, 
                      y > v.y ? y : v.y, 
                      z > v.z ? z : v.z); 
    }
};

FORCEINLINE Vector3 operator*(float s, const Vector3& v) { return v * s; }

struct ALIGN32 Ray {
    Vector3 origin;
    Vector3 direction;
    Vector3 inv_direction;
    int sign[3];
    
    FORCEINLINE Ray(const Vector3& o, const Vector3& d) : origin(o), direction(d.normalize()) {
        inv_direction = Vector3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
        sign[0] = (inv_direction.x < 0);
        sign[1] = (inv_direction.y < 0);
        sign[2] = (inv_direction.z < 0);
    }
    
    FORCEINLINE Vector3 at(float t) const { return origin + direction * t; }
};

struct ALIGN32 AABB {
    Vector3 min;
    Vector3 max;
    
    FORCEINLINE AABB() : min(FLT_MAX, FLT_MAX, FLT_MAX), max(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}
    FORCEINLINE AABB(const Vector3& min, const Vector3& max) : min(min), max(max) {}
    
    FORCEINLINE Vector3 center() const { return (min + max) * 0.5f; }
    
    FORCEINLINE bool intersect(const Ray& ray, float tmin, float tmax) const {
        float tx1 = (min.x - ray.origin.x) * ray.inv_direction.x;
        float tx2 = (max.x - ray.origin.x) * ray.inv_direction.x;
        
        tmin = (tx1 > tmin) ? tx1 : tmin;
        tmax = (tx2 < tmax) ? tx2 : tmax;
        if (tmax <= tmin) return false;
        
        float ty1 = (min.y - ray.origin.y) * ray.inv_direction.y;
        float ty2 = (max.y - ray.origin.y) * ray.inv_direction.y;
        
        tmin = (ty1 > tmin) ? ty1 : tmin;
        tmax = (ty2 < tmax) ? ty2 : tmax;
        if (tmax <= tmin) return false;
        
        float tz1 = (min.z - ray.origin.z) * ray.inv_direction.z;
        float tz2 = (max.z - ray.origin.z) * ray.inv_direction.z;
        
        tmin = (tz1 > tmin) ? tz1 : tmin;
        tmax = (tz2 < tmax) ? tz2 : tmax;
        return tmax > tmin;
    }
    
    FORCEINLINE static AABB surrounding(const AABB& a, const AABB& b) {
        return AABB(a.min.min(b.min), a.max.max(b.max));
    }
};

struct ALIGN32 Material {
    Vector3 albedo;
    float metallic;
    float roughness;
    Vector3 emission;
    float ior;
    
    FORCEINLINE Material() : albedo(0.8f, 0.8f, 0.8f), metallic(0.0f), roughness(0.5f), 
                           emission(0.0f, 0.0f, 0.0f), ior(1.5f) {}
};

struct ALIGN32 Sphere {
    Vector3 center;
    float radius;
    Material material;
    int object_id;
    AABB bbox;
    
    FORCEINLINE Sphere() : center(0, 0, 0), radius(1.0f), object_id(0) {
        update_bbox();
    }
    
    FORCEINLINE Sphere(const Vector3& c, float r, const Material& m, int id) 
        : center(c), radius(r), material(m), object_id(id) {
        update_bbox();
    }
    
    FORCEINLINE void update_bbox() {
        Vector3 rvec(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    }
    
    FORCEINLINE bool intersect(const Ray& ray, float tmin, float tmax, 
                              float& t, Vector3& normal, Material& mat, int& id) const {
        Vector3 oc = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float half_b = oc.dot(ray.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        
        if (discriminant < 0) return false;
        
        float sqrtd = sqrtf(discriminant);
        float root = (-half_b - sqrtd) / a;
        if (root < tmin || root > tmax) {
            root = (-half_b + sqrtd) / a;
            if (root < tmin || root > tmax) return false;
        }
        
        t = root;
        Vector3 hit_point = ray.at(t);
        normal = (hit_point - center) * (1.0f / radius);
        mat = material;
        id = object_id;
        return true;
    }
};

// ================================================
// FLATTENED BVH NODE (Array-based)
// ================================================
struct ALIGN32 BVHNodeFlat {
    AABB bbox;
    union {
        struct {
            int left_child;  // If primitive_count == 0, this is left child index
            int right_child; // If primitive_count == 0, this is right child index
        };
        struct {
            int first_primitive; // If primitive_count > 0, start index in primitive list
            int primitive_count; // Number of primitives in this leaf (0 for internal)
        };
    };
    
    FORCEINLINE bool is_leaf() const { return primitive_count > 0; }
};

// ================================================
// BVH BUILDER FORWARD DECLARATION
// ================================================
class SceneIntersector;

// ================================================
// OPTIMIZED CAMERA
// ================================================
struct Camera {
    Vector3 position;
    Vector3 forward;
    Vector3 right;
    Vector3 up;
    float fov;
    float aspect_ratio;
    float tan_fov;
    
    FORCEINLINE Camera() : position(0, 2, 3), fov(45.0f), aspect_ratio(1.333f) {
        update_basis();
    }
    
    FORCEINLINE void update_basis() {
        Vector3 target(0, 0, -3);
        forward = (target - position).normalize();
        right = forward.cross(Vector3(0, 1, 0)).normalize();
        up = right.cross(forward).normalize();
        tan_fov = tanf(fov * 3.14159f / 360.0f);
    }
    
    FORCEINLINE Ray get_ray(float u, float v) const {
        float view_x = (u - 0.5f) * aspect_ratio * tan_fov;
        float view_y = (0.5f - v) * tan_fov;
        Vector3 direction = forward + (right * view_x) + (up * view_y);
        return Ray(position, direction.normalize());
    }
    
    FORCEINLINE void move(const Vector3& delta) {
        position = position + delta;
        update_basis();
    }
};

// ================================================
// FAST MATH UTILITIES
// ================================================
namespace FastMath {
    FORCEINLINE float rsqrt(float x) {
        // Fast reciprocal sqrt (less accurate but faster)
        float xhalf = 0.5f * x;
        int i = *(int*)&x;
        i = 0x5f3759df - (i >> 1);
        x = *(float*)&i;
        x = x * (1.5f - xhalf * x * x);
        return x;
    }
    
    FORCEINLINE Vector3 reflect(const Vector3& v, const Vector3& n) {
        return v - n * (2.0f * v.dot(n));
    }
    
    FORCEINLINE bool refract(const Vector3& v, const Vector3& n, float ni_over_nt, Vector3& refracted) {
        Vector3 uv = v.normalize();
        float dt = uv.dot(n);
        float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
        if (discriminant > 0) {
            refracted = (uv - n * dt) * ni_over_nt - n * sqrtf(discriminant);
            return true;
        }
        return false;
    }
    
    FORCEINLINE float schlick(float cosine, float ref_idx) {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
    }
    
    FORCEINLINE Vector3 random_in_unit_sphere(PCG32& rng) {
        Vector3 p;
        do {
            p = Vector3(rng.random_float(-1.0f, 1.0f), 
                       rng.random_float(-1.0f, 1.0f), 
                       rng.random_float(-1.0f, 1.0f));
        } while (p.length_squared() >= 1.0f);
        return p;
    }
    
    FORCEINLINE Vector3 random_in_hemisphere(const Vector3& normal, PCG32& rng) {
        Vector3 in_unit_sphere = random_in_unit_sphere(rng);
        if (in_unit_sphere.dot(normal) > 0.0f) {
            return in_unit_sphere;
        }
        else {
            return in_unit_sphere * -1.0f;
        }
    }
}