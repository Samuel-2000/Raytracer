#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "raytracer_core.h"

namespace py = pybind11;

// Helper functions for operator overloading
Vector3 vector3_add(const Vector3& a, const Vector3& b) { return a + b; }
Vector3 vector3_sub(const Vector3& a, const Vector3& b) { return a - b; }
Vector3 vector3_mul_scalar(const Vector3& a, double scalar) { return a * scalar; }
Vector3 vector3_mul_vector(const Vector3& a, const Vector3& b) { return a * b; }
Vector3 vector3_div(const Vector3& a, double scalar) { return a / scalar; }
Vector3 vector3_neg(const Vector3& a) { return -a; }
Vector3& vector3_iadd(Vector3& a, const Vector3& b) { return a += b; }
Vector3& vector3_imul(Vector3& a, double scalar) { return a *= scalar; }

PYBIND11_MODULE(raytracer_cpp, m) {
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z)
        // Operators
        .def("__add__", &vector3_add)
        .def("__sub__", &vector3_sub)
        .def("__mul__", &vector3_mul_scalar)
        .def("__mul__", &vector3_mul_vector)
        .def("__rmul__", &vector3_mul_scalar)
        .def("__truediv__", &vector3_div)
        .def("__neg__", &vector3_neg)
        .def("__iadd__", &vector3_iadd)
        .def("__imul__", &vector3_imul)
        // Methods
        .def("dot", &Vector3::dot)
        .def("cross", &Vector3::cross)
        .def("length_squared", &Vector3::length_squared)
        .def("length", &Vector3::length)
        .def("normalize", &Vector3::normalize)
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    py::class_<Ray>(m, "Ray")
        .def(py::init<const Vector3&, const Vector3&>())
        .def_readwrite("origin", &Ray::origin)
        .def_readwrite("direction", &Ray::direction)
        .def("at", &Ray::at);
    
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("albedo", &Material::albedo)
        .def_readwrite("metallic", &Material::metallic)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("emission", &Material::emission)
        .def_readwrite("ior", &Material::ior);
    
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("material", &Sphere::material)
        .def_readwrite("object_id", &Sphere::object_id)
        .def_readwrite("name", &Sphere::name)
        .def("hit", &Sphere::hit);
    
    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("position", &Camera::position)
        .def_readwrite("target", &Camera::target)
        .def_readwrite("up", &Camera::up)
        .def_readwrite("fov", &Camera::fov)
        .def_readwrite("aspect_ratio", &Camera::aspect_ratio)
        .def("get_ray", &Camera::get_ray)
        .def("move", &Camera::move)
        .def("rotate", &Camera::rotate);

    py::class_<DebugInfo>(m, "DebugInfo")
        .def_readwrite("enable_debug", &DebugInfo::enable_debug)
        .def_readwrite("build_count", &DebugInfo::build_count)
        .def_readwrite("render_count", &DebugInfo::render_count)
        .def("reset", &DebugInfo::reset)
        .def("get_stats", &DebugInfo::get_stats);
    
    py::class_<Scene>(m, "Scene")
        .def(py::init<>())
        .def_readwrite("spheres", &Scene::spheres)
        .def_readwrite("background_color", &Scene::background_color)
        .def_readwrite("use_bvh", &Scene::use_bvh)
        .def_readwrite("debug_mode", &Scene::debug_mode)
        .def("add_sphere", &Scene::add_sphere)
        .def("remove_sphere", &Scene::remove_sphere)
        .def("build_bvh", &Scene::build_bvh)
        .def("hit", &Scene::hit)
        .def("cast_ray_for_selection", &Scene::cast_ray_for_selection);
    
    py::class_<RayTracer>(m, "RayTracer")
        .def(py::init<>())
        .def("set_scene", &RayTracer::set_scene)
        .def("render", &RayTracer::render)
        .def("get_camera", &RayTracer::get_camera_copy)
        .def("set_camera", &RayTracer::set_camera)
        .def("get_camera", &RayTracer::get_camera, py::return_value_policy::reference)
        .def("select_object", &RayTracer::select_object)
        .def("move_camera", &RayTracer::move_camera)
        .def("trace_ray", &RayTracer::trace_ray)
        .def("set_debug_mode", &RayTracer::set_debug_mode)
        .def("get_debug_info", &RayTracer::get_debug_info);
}