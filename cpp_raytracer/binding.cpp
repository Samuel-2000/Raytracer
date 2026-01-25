// binding.cpp - UPDATED TO INCLUDE MaterialType enum
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "raytracer_core.h"
#include "textures.h"
#include "vector3.h"
#include "material.h"

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
        .def("__add__", &vector3_add)
        .def("__sub__", &vector3_sub)
        .def("__mul__", &vector3_mul_scalar)
        .def("__mul__", &vector3_mul_vector)
        .def("__rmul__", &vector3_mul_scalar)
        .def("__truediv__", &vector3_div)
        .def("__neg__", &vector3_neg)
        .def("__iadd__", &vector3_iadd)
        .def("__imul__", &vector3_imul)
        .def("dot", &Vector3::dot)
        .def("cross", &Vector3::cross)
        .def("length_squared", &Vector3::length_squared)
        .def("length", &Vector3::length)
        .def("normalize", &Vector3::normalize)
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    // MaterialType enum - ADD THIS SECTION
    py::enum_<MaterialType>(m, "MaterialType")
        .value("CUSTOM", MATERIAL_CUSTOM)
        .value("DIFFUSE", MATERIAL_DIFFUSE)
        .value("METAL", MATERIAL_METAL)
        .value("DIELECTRIC", MATERIAL_DIELECTRIC)
        .value("PLASTIC", MATERIAL_PLASTIC)
        .value("WOOD", MATERIAL_WOOD)
        .value("MARBLE", MATERIAL_MARBLE)
        .value("RUSTY_METAL", MATERIAL_RUSTY_METAL)
        .value("GLASS", MATERIAL_GLASS)
        .value("MIRROR", MATERIAL_MIRROR)
        .value("RUBBER", MATERIAL_RUBBER)
        .export_values();
    
    // Texture base class
    py::class_<Texture, std::shared_ptr<Texture>>(m, "Texture")
        .def("value", &Texture::value);
    
    // Texture implementations
    py::class_<SolidTexture, Texture, std::shared_ptr<SolidTexture>>(m, "SolidTexture")
        .def(py::init<const Vector3&>());
    
    py::class_<NoiseTexture, Texture, std::shared_ptr<NoiseTexture>>(m, "NoiseTexture")
        .def(py::init<float>());
    
    py::class_<CheckerTexture, Texture, std::shared_ptr<CheckerTexture>>(m, "CheckerTexture")
        .def(py::init<const Vector3&, const Vector3&, float>());
    
    py::class_<ImageTexture, Texture, std::shared_ptr<ImageTexture>>(m, "ImageTexture")
        .def(py::init<>())
        .def(py::init<const char*>())
        .def("load", &ImageTexture::load);
    
    py::class_<WoodTexture, Texture, std::shared_ptr<WoodTexture>>(m, "WoodTexture")
        .def(py::init<float, const Vector3&>(), py::arg("scale") = 5.0f, py::arg("base_color") = Vector3(0.6,0.4,0.2));
    
    py::class_<MarbleTexture, Texture, std::shared_ptr<MarbleTexture>>(m, "MarbleTexture")
        .def(py::init<float, const Vector3&>(), py::arg("scale") = 3.0f, py::arg("base_color") = Vector3(0.8,0.8,0.9));
    
    py::class_<MetalTexture, Texture, std::shared_ptr<MetalTexture>>(m, "MetalTexture")
        .def(py::init<float>(), py::arg("roughness_variation") = 0.1f)
        .def("roughness_value", &MetalTexture::roughness_value);
    
    // SkyboxType enum
    py::enum_<SkyboxType>(m, "SkyboxType")
        .value("SOLID", SkyboxType::SOLID)
        .value("GRADIENT", SkyboxType::GRADIENT)
        .value("SUNSET", SkyboxType::SUNSET)
        .value("ATMOSPHERE", SkyboxType::ATMOSPHERE)
        .value("IMAGE", SkyboxType::IMAGE)
        .export_values();
    
    // Skybox class
    py::class_<Skybox>(m, "Skybox")
        .def(py::init<>())
        .def("set_type", &Skybox::set_type)
        .def("set_colors", [](Skybox& skybox, const Vector3& c1, const Vector3& c2, const Vector3& c3) {
            skybox.set_colors(c1, c2, c3);
        }, py::arg("c1"), py::arg("c2") = Vector3(0,0,0), py::arg("c3") = Vector3(0,0,0))
        .def("set_atmosphere_colors", &Skybox::set_atmosphere_colors)
        .def("load_image", &Skybox::load_image)
        .def("get_color", &Skybox::get_color);
    
    // Material class
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("albedo", &Material::albedo)
        .def_readwrite("metallic", &Material::metallic)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("emission", &Material::emission)
        .def_readwrite("ior", &Material::ior)
        .def_readwrite("albedo_texture", &Material::albedo_texture)
        .def_readwrite("roughness_texture", &Material::roughness_texture)
        .def_readwrite("material_type", &Material::material_type);
    
    // Ray
    py::class_<Ray>(m, "Ray")
        .def(py::init<const Vector3&, const Vector3&>())
        .def_readwrite("origin", &Ray::origin)
        .def_readwrite("direction", &Ray::direction)
        .def("at", &Ray::at);
    
    // Sphere
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("material", &Sphere::material)
        .def_readwrite("object_id", &Sphere::object_id)
        .def_readwrite("name", &Sphere::name)
        .def("hit", &Sphere::hit);
    
    // Camera
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
    
    // Scene
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
        .def("cast_ray_for_selection", &Scene::cast_ray_for_selection)
        .def("set_skybox", &Scene::set_skybox)
        .def("get_skybox", &Scene::get_skybox);
    
    // RayTracer
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
        .def("get_scene", &RayTracer::get_scene, py::return_value_policy::reference);
}