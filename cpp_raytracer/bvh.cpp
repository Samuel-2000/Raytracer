// bvh.cpp
#define _USE_MATH_DEFINES
#include "bvh.h"
#include "raytracer_core.h"
#include <algorithm>
#include <cmath>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==================== AABB ====================
bool AABB::hit(const Ray& ray, double tmin, double tmax) const {
    for (int a = 0; a < 3; ++a) {
        double invD = 1.0 / ray.direction[a];
        double t0 = (min[a] - ray.origin[a]) * invD;
        double t1 = (max[a] - ray.origin[a]) * invD;
        if (invD < 0.0) std::swap(t0, t1);
        tmin = (t0 > tmin) ? t0 : tmin;
        tmax = (t1 < tmax) ? t1 : tmax;
        if (tmax <= tmin) return false;
    }
    return true;
}

int AABB::hit_packet(const RayPacket& packet, double tmin, double tmax, int active_mask) const {
    int result = 0;
    for (int i = 0; i < 4; ++i) {
        if (active_mask & (1 << i)) {
            Ray r(packet.origins[i], packet.directions[i]);
            if (hit(r, tmin, tmax)) result |= (1 << i);
        }
    }
    return result;
}

AABB AABB::surrounding_box(const AABB& box0, const AABB& box1) {
    return AABB(Vector3(std::fmin(box0.min.x, box1.min.x),
                        std::fmin(box0.min.y, box1.min.y),
                        std::fmin(box0.min.z, box1.min.z)),
                Vector3(std::fmax(box0.max.x, box1.max.x),
                        std::fmax(box0.max.y, box1.max.y),
                        std::fmax(box0.max.z, box1.max.z)));
}

AABB sphere_bounding_box(const Sphere& sphere) {
    Vector3 r(sphere.radius, sphere.radius, sphere.radius);
    return AABB(sphere.center - r, sphere.center + r);
}

// ==================== BVHNode ====================
BVHNode::BVHNode() : left(nullptr), right(nullptr), is_leaf(false) {}
BVHNode::~BVHNode() { delete left; delete right; }

#ifdef __AVX2__
#include <immintrin.h>

// SIMD kernel: test packet of 4 rays against one sphere, update t_max and hit records
static int packet_sphere_intersect(const RayPacket& packet, const Sphere& sphere,
                                   double t_min, double t_max_arr[4], HitRecord rec[4]) {
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
    int mask = _mm256_movemask_pd(mask_ge);
    if (mask == 0) return 0;
    
    __m256d sqrt_disc = _mm256_sqrt_pd(disc);
    __m256d t1 = _mm256_sub_pd(_mm256_sub_pd(zero, half_b), sqrt_disc);
    double t_vals[4];
    _mm256_store_pd(t_vals, t1);
    
    int hit_mask = 0;
    for (int i = 0; i < 4; ++i) {
        if (mask & (1 << i)) {
            double t = t_vals[i];
            if (t > t_min && t < t_max_arr[i]) {
                Ray ray(packet.origins[i], packet.directions[i]);
                HitRecord temp;
                if (sphere.hit(ray, t_min, t_max_arr[i], temp)) {
                    if (temp.t < t_max_arr[i]) {
                        t_max_arr[i] = temp.t;
                        rec[i] = temp;
                        hit_mask |= (1 << i);
                    }
                }
            }
        }
    }
    return hit_mask;
}
#endif

int BVHNode::hit_packet(const RayPacket& packet, double t_min, double t_max,
                        HitRecord rec[4], const std::vector<Sphere>& scene_spheres,
                        bool use_simd, int active_mask) const {
    // AABB test for all active rays
    int box_mask = box.hit_packet(packet, t_min, t_max, active_mask);
    if (box_mask == 0) return 0;
    
    if (is_leaf) {
#ifdef __AVX2__
        if (use_simd && sphere_indices.size() >= 4) {
            double t_max_arr[4] = {t_max, t_max, t_max, t_max};
            int result_mask = 0;
            for (int idx : sphere_indices) {
                int hit = packet_sphere_intersect(packet, scene_spheres[idx],
                                                  t_min, t_max_arr, rec);
                result_mask |= hit;
                if (result_mask == box_mask) break; // all active rays have a hit
            }
            return result_mask;
        }
#endif
        // Scalar leaf: test each sphere against each active ray
        int result_mask = 0;
        for (int idx : sphere_indices) {
            const Sphere& sphere = scene_spheres[idx];
            for (int i = 0; i < 4; ++i) {
                if (active_mask & (1 << i)) {
                    Ray r(packet.origins[i], packet.directions[i]);
                    if (sphere.hit(r, t_min, t_max, rec[i])) {
                        result_mask |= (1 << i);
                    }
                }
            }
        }
        return result_mask;
    }
    
    // Internal node: recurse
    int left_mask = left ? left->hit_packet(packet, t_min, t_max, rec, scene_spheres, use_simd, box_mask) : 0;
    int right_mask = right ? right->hit_packet(packet, t_min, t_max, rec, scene_spheres, use_simd, box_mask & ~left_mask) : 0;
    return left_mask | right_mask;
}

bool BVHNode::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
                  const std::vector<Sphere>& scene_spheres, bool use_simd) const {
    if (!box.hit(ray, t_min, t_max)) return false;
    if (is_leaf) {
        bool hit_any = false;
        double closest = t_max;
        for (int idx : sphere_indices) {
            HitRecord temp;
            if (scene_spheres[idx].hit(ray, t_min, closest, temp)) {
                hit_any = true;
                closest = temp.t;
                rec = temp;
            }
        }
        return hit_any;
    }
    HitRecord left_rec, right_rec;
    bool hit_left = left && left->hit(ray, t_min, t_max, left_rec, scene_spheres, use_simd);
    bool hit_right = right && right->hit(ray, t_min, t_max, right_rec, scene_spheres, use_simd);
    if (hit_left && hit_right) {
        rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
        return true;
    } else if (hit_left) {
        rec = left_rec;
        return true;
    } else if (hit_right) {
        rec = right_rec;
        return true;
    }
    return false;
}

// ==================== BVH ====================
bool BVH::box_compare(const Sphere& a, const Sphere& b, int axis) {
    AABB ba = sphere_bounding_box(a);
    AABB bb = sphere_bounding_box(b);
    if (axis == 0) return ba.min.x < bb.min.x;
    if (axis == 1) return ba.min.y < bb.min.y;
    return ba.min.z < bb.min.z;
}

BVHNode* BVH::build_tree(const std::vector<Sphere>& scene_spheres,
                         std::vector<int>& indices, size_t start, size_t end,
                         int depth, bool debug_mode) {
    if (start >= end) return nullptr;
    BVHNode* node = new BVHNode();
    node_count++;
    size_t span = end - start;
    if (span <= 4) {
        for (size_t i = start; i < end; ++i) node->sphere_indices.push_back(indices[i]);
        node->is_leaf = true;
        int first = node->sphere_indices[0];
        node->box = sphere_bounding_box(scene_spheres[first]);
        for (size_t i = 1; i < node->sphere_indices.size(); ++i) {
            int idx = node->sphere_indices[i];
            node->box = AABB::surrounding_box(node->box, sphere_bounding_box(scene_spheres[idx]));
        }
        return node;
    }
    // compute total bounding box
    int first = indices[start];
    AABB total = sphere_bounding_box(scene_spheres[first]);
    for (size_t i = start+1; i < end; ++i) {
        total = AABB::surrounding_box(total, sphere_bounding_box(scene_spheres[indices[i]]));
    }
    node->box = total;
    Vector3 extent = total.max - total.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.y && extent.z > extent.x) axis = 2;
    
    auto comp = [axis, &scene_spheres, this](int ia, int ib) {
        return box_compare(scene_spheres[ia], scene_spheres[ib], axis);
    };
    std::sort(indices.begin() + start, indices.begin() + end, comp);
    size_t mid = start + span / 2;
    node->left = build_tree(scene_spheres, indices, start, mid, depth+1, debug_mode);
    node->right = build_tree(scene_spheres, indices, mid, end, depth+1, debug_mode);
    node->is_leaf = false;
    return node;
}

BVH::BVH() : root(nullptr), node_count(0) {}
BVH::~BVH() { delete root; }

void BVH::build(const std::vector<Sphere>& scene_spheres, bool debug_mode) {
    if (scene_spheres.empty()) return;
    delete root;
    root = nullptr;
    node_count = 0;
    std::vector<int> indices(scene_spheres.size());
    for (size_t i = 0; i < scene_spheres.size(); ++i) indices[i] = i;
    root = build_tree(scene_spheres, indices, 0, indices.size(), 0, debug_mode);
}

void BVH::refit(const std::vector<Sphere>& scene_spheres) {
    if (!root) return;
    std::function<void(BVHNode*)> refit_node = [&](BVHNode* node) {
        if (!node) return;
        if (node->is_leaf) {
            if (!node->sphere_indices.empty()) {
                int first = node->sphere_indices[0];
                node->box = sphere_bounding_box(scene_spheres[first]);
                for (size_t i = 1; i < node->sphere_indices.size(); ++i) {
                    int idx = node->sphere_indices[i];
                    node->box = AABB::surrounding_box(node->box, sphere_bounding_box(scene_spheres[idx]));
                }
            }
        } else {
            refit_node(node->left);
            refit_node(node->right);
            if (node->left && node->right)
                node->box = AABB::surrounding_box(node->left->box, node->right->box);
            else if (node->left)
                node->box = node->left->box;
            else if (node->right)
                node->box = node->right->box;
        }
    };
    refit_node(root);
}

bool BVH::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec,
             const std::vector<Sphere>& scene_spheres, bool use_simd) const {
    if (!root) return false;
    return root->hit(ray, t_min, t_max, rec, scene_spheres, use_simd);
}

int BVH::hit_packet(const RayPacket& packet, double t_min, double t_max,
                    HitRecord rec[4], const std::vector<Sphere>& scene_spheres,
                    bool use_simd) const {
    if (!root) return 0;
    return root->hit_packet(packet, t_min, t_max, rec, scene_spheres, use_simd, 0xF);
}