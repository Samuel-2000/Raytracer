// raytracer_core.cpp - FIXED SIMD packet alignment
#define _USE_MATH_DEFINES
#include "raytracer_core.h"
#include "bvh.h"
#include "textures.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

thread_local std::mt19937 thread_local_gen;
thread_local std::uniform_real_distribution<double> thread_local_dis(0.0, 1.0);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Ray Camera::get_ray(double u, double v) const {
    double ndc_x = (u - 0.5) * 2.0;
    double ndc_y = (0.5 - v) * 2.0;
    Vector3 forward = (target - position).normalize();
    Vector3 right = forward.cross(Vector3(0,1,0)).normalize();
    if (right.length() < 0.001) right = Vector3(1,0,0);
    Vector3 up = right.cross(forward).normalize();
    double fov_rad = fov * M_PI / 180.0;
    double viewport_height = std::tan(fov_rad / 2.0);
    double viewport_width = viewport_height * aspect_ratio;
    Vector3 dir = forward + (right * (ndc_x * viewport_width)) + (up * (ndc_y * viewport_height));
    return Ray(position, dir.normalize());
}

bool Sphere::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    Vector3 oc = ray.origin - center;
    double a = ray.direction.dot(ray.direction);
    double half_b = oc.dot(ray.direction);
    double c = oc.dot(oc) - radius*radius;
    double disc = half_b*half_b - a*c;
    if (disc < 0) return false;
    double sqrtd = std::sqrt(disc);
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) return false;
    }
    rec.t = root;
    rec.point = ray.at(rec.t);
    Vector3 outward_normal = (rec.point - center) * (1.0 / radius);
    rec.set_face_normal(ray, outward_normal);
    rec.albedo = material.albedo;
    rec.metallic = material.metallic;
    rec.roughness = material.roughness;
    rec.emission = material.emission;
    rec.ior = material.ior;
    rec.material_albedo_texture = material.albedo_texture;
    rec.material_roughness_texture = material.roughness_texture;
    rec.sphere_center = center;
    rec.object_id = object_id;
    return true;
}

// ==================== Scene ====================
Scene::Scene()
    : background_color(0.1,0.1,0.1), bvh(nullptr), use_bvh(true),
      dynamic_bvh(false), simd_ray_hit(false), debug_mode(false), skybox(nullptr) {}
Scene::Scene(const Scene& other)
    : spheres(other.spheres), background_color(other.background_color), bvh(nullptr),
      use_bvh(other.use_bvh), dynamic_bvh(other.dynamic_bvh), simd_ray_hit(other.simd_ray_hit),
      debug_mode(other.debug_mode), skybox(other.skybox) {}
Scene::~Scene() { delete bvh; }

Scene& Scene::operator=(const Scene& other) {
    if (this == &other) return *this;
    delete bvh; bvh = nullptr;
    spheres = other.spheres;
    background_color = other.background_color;
    use_bvh = other.use_bvh;
    dynamic_bvh = other.dynamic_bvh;
    simd_ray_hit = other.simd_ray_hit;
    debug_mode = other.debug_mode;
    skybox = other.skybox;
    if (use_bvh) build_bvh();
    return *this;
}

void Scene::set_skybox(std::shared_ptr<Skybox> new_skybox) { skybox = std::move(new_skybox); }
std::shared_ptr<Skybox> Scene::get_skybox() const { return skybox; }
void Scene::add_sphere(const Sphere& sphere) { spheres.push_back(sphere); }
void Scene::remove_sphere(int id) {
    spheres.erase(std::remove_if(spheres.begin(), spheres.end(),
        [id](const Sphere& s) { return s.object_id == id; }), spheres.end());
}
void Scene::build_bvh() { if (bvh) delete bvh; bvh = new BVH(); bvh->build(spheres, debug_mode); }
void Scene::refit_bvh() { if (use_bvh && bvh && dynamic_bvh) bvh->refit(spheres); }

bool Scene::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (use_bvh && bvh) return bvh->hit(ray, t_min, t_max, rec, spheres, simd_ray_hit);
    bool hit_any = false;
    double closest = t_max;
    HitRecord temp;
    for (const auto& s : spheres) {
        if (s.hit(ray, t_min, closest, temp)) {
            hit_any = true;
            closest = temp.t;
            rec = temp;
        }
    }
    return hit_any;
}

#ifdef __AVX2__
static int linear_hit_packet_simd(const RayPacket& packet, double t_min, double t_max,
                                  const std::vector<Sphere>& spheres, HitRecord rec[4],
                                  int active_mask) {
    int result_mask = 0;
    double t_max_arr[4] = {t_max, t_max, t_max, t_max};
    for (const auto& sphere : spheres) {
        if ((result_mask & active_mask) == active_mask) break;

        __m256d cx = _mm256_set1_pd(sphere.center.x);
        __m256d cy = _mm256_set1_pd(sphere.center.y);
        __m256d cz = _mm256_set1_pd(sphere.center.z);
        __m256d r = _mm256_set1_pd(sphere.radius);

        __m256d ox = _mm256_set_pd(packet.origins[3].x, packet.origins[2].x,
                                   packet.origins[1].x, packet.origins[0].x);
        __m256d oy = _mm256_set_pd(packet.origins[3].y, packet.origins[2].y,
                                   packet.origins[1].y, packet.origins[0].y);
        __m256d oz = _mm256_set_pd(packet.origins[3].z, packet.origins[2].z,
                                   packet.origins[1].z, packet.origins[0].z);
        __m256d dx = _mm256_set_pd(packet.directions[3].x, packet.directions[2].x,
                                   packet.directions[1].x, packet.directions[0].x);
        __m256d dy = _mm256_set_pd(packet.directions[3].y, packet.directions[2].y,
                                   packet.directions[1].y, packet.directions[0].y);
        __m256d dz = _mm256_set_pd(packet.directions[3].z, packet.directions[2].z,
                                   packet.directions[1].z, packet.directions[0].z);

        __m256d ocx = _mm256_sub_pd(ox, cx);
        __m256d ocy = _mm256_sub_pd(oy, cy);
        __m256d ocz = _mm256_sub_pd(oz, cz);

        __m256d half_b = _mm256_add_pd(_mm256_mul_pd(ocx, dx),
                                       _mm256_add_pd(_mm256_mul_pd(ocy, dy),
                                                     _mm256_mul_pd(ocz, dz)));
        __m256d oc_len2 = _mm256_add_pd(_mm256_mul_pd(ocx, ocx),
                                        _mm256_add_pd(_mm256_mul_pd(ocy, ocy),
                                                      _mm256_mul_pd(ocz, ocz)));
        __m256d c = _mm256_sub_pd(oc_len2, _mm256_mul_pd(r, r));
        __m256d disc = _mm256_sub_pd(_mm256_mul_pd(half_b, half_b), c);
        __m256d zero = _mm256_set1_pd(0.0);
        __m256d mask_ge = _mm256_cmp_pd(disc, zero, _CMP_GE_OQ);
        int mask = _mm256_movemask_pd(mask_ge) & active_mask;
        if (mask == 0) continue;

        __m256d sqrt_disc = _mm256_sqrt_pd(disc);
        __m256d t1 = _mm256_sub_pd(_mm256_sub_pd(zero, half_b), sqrt_disc);
        __m256d t2 = _mm256_add_pd(_mm256_sub_pd(zero, half_b), sqrt_disc);

        double t1_vals[4], t2_vals[4];
        _mm256_storeu_pd(t1_vals, t1);
        _mm256_storeu_pd(t2_vals, t2);

        for (int i = 0; i < 4; ++i) {
            if (!(mask & (1 << i))) continue;

            double t = 1e20;
            if (t1_vals[i] > t_min && t1_vals[i] < t_max_arr[i]) t = t1_vals[i];
            if (t2_vals[i] > t_min && t2_vals[i] < t_max_arr[i] && t2_vals[i] < t) t = t2_vals[i];

            if (t < t_max_arr[i]) {
                Ray ray(packet.origins[i], packet.directions[i]);
                HitRecord temp;
                if (sphere.hit(ray, t_min, t_max_arr[i], temp)) {
                    if (temp.t < t_max_arr[i]) {
                        t_max_arr[i] = temp.t;
                        rec[i] = temp;
                        result_mask |= (1 << i);
                    }
                }
            }
        }
    }
    return result_mask;
}
#endif

int Scene::hit_packet(const RayPacket& packet, HitRecord rec[4], int active_mask) const {
    // Branch 1: SIMD + BVH
    if (simd_ray_hit && use_bvh && bvh != nullptr) {
        return bvh->hit_packet(packet, 0.001, 1e10, rec, spheres, true, active_mask);
    }
#ifdef __AVX2__
    // Branch 2: SIMD + linear search (BVH off, SIMD on)
    if (simd_ray_hit) {
        return linear_hit_packet_simd(packet, 0.001, 1e10, spheres, rec, active_mask);
    }
#endif
    // Branch 3: scalar fallback – only process active rays
    int mask = 0;
    for (int i = 0; i < 4; ++i) {
        if (!(active_mask & (1 << i))) continue;
        Ray r(packet.origins[i], packet.directions[i]);
        if (hit(r, 0.001, 1e10, rec[i])) mask |= (1 << i);
    }
    return mask;
}

int Scene::cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const {
    HitRecord rec;
    int selected = -1;
    double closest = t_max;
    for (const auto& s : spheres) {
        if (s.hit(ray, t_min, closest, rec)) {
            closest = rec.t;
            selected = s.object_id;
        }
    }
    return selected;
}

// ==================== RayTracer ====================
RayTracer::RayTracer() : gen(std::random_device{}()), dis(0.0,1.0) {
    #ifdef _OPENMP
    #pragma omp parallel
    { thread_local_gen = std::mt19937(std::random_device{}() + omp_get_thread_num()); }
    #else
    thread_local_gen = std::mt19937(std::random_device{}());
    #endif
}
RayTracer::~RayTracer() {}

Vector3 RayTracer::sample_albedo(const HitRecord& rec) const {
    if (rec.material_albedo_texture) {
        Vector3 p = rec.point - rec.sphere_center;
        double phi = atan2(p.z, p.x);
        double theta = asin(p.y / rec.sphere_center.length());
        double u = 0.5 + phi / (2*M_PI);
        double v = 0.5 + theta / M_PI;
        return rec.material_albedo_texture->value(u, v, rec.point);
    }
    return rec.albedo;
}
float RayTracer::sample_roughness(const HitRecord& rec) const {
    if (rec.material_roughness_texture) {
        Vector3 p = rec.point - rec.sphere_center;
        double phi = atan2(p.z, p.x);
        double theta = asin(p.y / rec.sphere_center.length());
        double u = 0.5 + phi / (2*M_PI);
        double v = 0.5 + theta / M_PI;
        return rec.material_roughness_texture->roughness_value(u, v, rec.point);
    }
    return static_cast<float>(rec.roughness);
}

void RayTracer::set_scene(const Scene& new_scene) {
    scene = new_scene;
    if (scene.use_bvh) scene.build_bvh();
}

Vector3 RayTracer::random_in_unit_sphere() {
    Vector3 p;
    do { p = Vector3(thread_local_dis(thread_local_gen), thread_local_dis(thread_local_gen), thread_local_dis(thread_local_gen)) * 2.0 - Vector3(1,1,1); }
    while (p.length_squared() >= 1.0);
    return p;
}
Vector3 RayTracer::random_in_hemisphere(const Vector3& normal) {
    Vector3 in_unit = random_in_unit_sphere();
    return (in_unit.dot(normal) > 0.0) ? in_unit : in_unit * -1.0;
}
Vector3 RayTracer::reflect(const Vector3& v, const Vector3& n) { return v - n * (2.0 * v.dot(n)); }
bool RayTracer::refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted) {
    Vector3 uv = v.normalize();
    double dt = uv.dot(n);
    double disc = 1.0 - ni_over_nt * ni_over_nt * (1 - dt*dt);
    if (disc > 0) { refracted = (uv - n*dt) * ni_over_nt - n * std::sqrt(disc); return true; }
    return false;
}
double RayTracer::schlick(double cosine, double ref_idx) {
    double r0 = (1-ref_idx)/(1+ref_idx); r0 = r0*r0;
    return r0 + (1-r0)*std::pow(1-cosine,5);
}

Vector3 RayTracer::trace_ray(const Ray& ray, int depth, int max_depth) {
    if (depth <= 0) return Vector3(0,0,0);
    HitRecord rec;
    if (scene.hit(ray, 0.001, 1e10, rec)) {
        Vector3 emitted = rec.emission;
        Vector3 albedo = sample_albedo(rec);
        float roughness = sample_roughness(rec);
        double prob = 0.8;
        int bounces = max_depth - depth;
        if (bounces < 3 || thread_local_dis(thread_local_gen) < prob) {
            Vector3 color;
            if (thread_local_dis(thread_local_gen) < rec.metallic) {
                Vector3 reflected = reflect(ray.direction.normalize(), rec.normal);
                Vector3 scatter = random_in_unit_sphere() * roughness;
                Ray scattered(rec.point, reflected + scatter);
                color = trace_ray(scattered, depth-1, max_depth);
            } else {
                Vector3 target = rec.point + rec.normal + random_in_hemisphere(rec.normal);
                Ray scattered(rec.point, target - rec.point);
                color = trace_ray(scattered, depth-1, max_depth);
            }
            return emitted + (color * albedo) / prob;
        }
        return emitted;
    }
    if (scene.skybox) return scene.skybox->get_color(ray.direction);
    return scene.background_color;
}

void RayTracer::trace_packet(const RayPacket& packet, Vector3 colors[4], int depth, int max_depth, int active_mask) {
    if (depth <= 0) {
        for (int i = 0; i < 4; ++i) colors[i] = Vector3(0,0,0);
        return;
    }

    HitRecord rec[4];
    int hit_mask = scene.hit_packet(packet, rec, active_mask);

    for (int i = 0; i < 4; ++i) {
        if (!(active_mask & (1 << i))) {
            colors[i] = Vector3(0,0,0);
            continue;
        }
        if (!(hit_mask & (1 << i))) {
            if (scene.skybox)
                colors[i] = scene.skybox->get_color(packet.directions[i]);
            else
                colors[i] = scene.background_color;
        } else {
            Vector3 emitted = rec[i].emission;
            Vector3 albedo = sample_albedo(rec[i]);
            float roughness = sample_roughness(rec[i]);
            double prob = 0.8;
            int bounces = max_depth - depth;
            Vector3 color;
            if (bounces < 3 || thread_local_dis(thread_local_gen) < prob) {
                if (thread_local_dis(thread_local_gen) < rec[i].metallic) {
                    Vector3 reflected = reflect(packet.directions[i].normalize(), rec[i].normal);
                    Vector3 scatter = random_in_unit_sphere() * roughness;
                    Ray scattered(rec[i].point, reflected + scatter);
                    color = trace_ray(scattered, depth-1, max_depth);
                } else {
                    Vector3 target = rec[i].point + rec[i].normal + random_in_hemisphere(rec[i].normal);
                    Ray scattered(rec[i].point, target - rec[i].point);
                    color = trace_ray(scattered, depth-1, max_depth);
                }
                colors[i] = emitted + (color * albedo) / prob;
            } else {
                colors[i] = emitted;
            }
        }
    }
}

std::vector<double> RayTracer::render(int width, int height, int samples_per_pixel, int max_depth) {
    camera.aspect_ratio = double(width)/height;
    std::vector<double> image_data(width * height * 3, 0.0);
    auto start = std::chrono::high_resolution_clock::now();
    const int TILE_SIZE = 32;
    const int PACKET_SIZE = 2;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int tile_y = 0; tile_y < height; tile_y += TILE_SIZE) {
        for (int tile_x = 0; tile_x < width; tile_x += TILE_SIZE) {
            int tile_end_y = std::min(tile_y + TILE_SIZE, height);
            int tile_end_x = std::min(tile_x + TILE_SIZE, width);
            for (int j = tile_y; j < tile_end_y; j += PACKET_SIZE) {
                int j_end = std::min(j + PACKET_SIZE, tile_end_y);
                for (int i = tile_x; i < tile_end_x; i += PACKET_SIZE) {
                    int i_end = std::min(i + PACKET_SIZE, tile_end_x);
                    
                    // Build packet and active mask
                    RayPacket packet;
                    int active_mask = 0;
                    int idx = 0;
                    for (int dy = 0; dy < j_end - j; ++dy) {
                        for (int dx = 0; dx < i_end - i; ++dx) {
                            double u = double(i + dx) / width;
                            double v = double(j + dy) / height;
                            Ray r = camera.get_ray(u, v);
                            packet.origins[idx] = r.origin;
                            packet.directions[idx] = r.direction;
                            active_mask |= (1 << idx);
                            ++idx;
                        }
                    }
                    // Fill remaining slots with dummy values (still needed for alignment)
                    while (idx < 4) {
                        packet.origins[idx] = Vector3(0,0,0);
                        packet.directions[idx] = Vector3(0,0,0);
                        ++idx;
                    }
                    
                    Vector3 acc_colors[4] = {Vector3(0,0,0), Vector3(0,0,0), Vector3(0,0,0), Vector3(0,0,0)};
                    for (int s = 0; s < samples_per_pixel; ++s) {
                        Vector3 colors[4];
                        trace_packet(packet, colors, max_depth, max_depth, active_mask);
                        for (int r = 0; r < 4; ++r) {
                            if (active_mask & (1 << r))
                                acc_colors[r] = acc_colors[r] + colors[r];
                        }
                    }
                    idx = 0;
                    for (int dy = 0; dy < j_end - j; ++dy) {
                        for (int dx = 0; dx < i_end - i; ++dx) {
                            Vector3 color = acc_colors[idx] * (1.0 / samples_per_pixel);
                            color = Vector3(std::sqrt(color.x), std::sqrt(color.y), std::sqrt(color.z));
                            int px = i + dx;
                            int py = j + dy;
                            int pixel_idx = (py * width + px) * 3;
                            image_data[pixel_idx] = color.x;
                            image_data[pixel_idx+1] = color.y;
                            image_data[pixel_idx+2] = color.z;
                            ++idx;
                        }
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Render time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return image_data;
}

int RayTracer::select_object(double x, double y, int width, int height) {
    Ray ray = camera.get_ray(x, y);
    return scene.cast_ray_for_selection(ray, 0.001, 1000.0);
}
void RayTracer::move_camera(const Vector3& delta) { camera.move(delta); }