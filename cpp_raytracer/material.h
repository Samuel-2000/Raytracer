// material.h
#pragma once
#include "vector3.h"
#include <memory>

// Forward declaration
class Texture;

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