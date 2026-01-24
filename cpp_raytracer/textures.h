#pragma once
#include "raytracer_core.h"
#include <memory>
#include <string>

// ==================== Texture Base Class ====================
class Texture {
public:
    virtual ~Texture() = default;
    virtual Vector3 value(double u, double v, const Vector3& p) const = 0;
    virtual float roughness_value(double u, double v, const Vector3& p) const { return 0.5f; }
};

// ==================== Solid Color Texture ====================
class SolidTexture : public Texture {
private:
    Vector3 color;
    
public:
    SolidTexture(const Vector3& c) : color(c) {}
    virtual Vector3 value(double u, double v, const Vector3& p) const override {
        return color;
    }
};

// ==================== Noise Texture ====================
class NoiseTexture : public Texture {
private:
    float scale;
    
public:
    NoiseTexture(float scale = 1.0f);
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
};

// ==================== Checker Texture ====================
class CheckerTexture : public Texture {
private:
    Vector3 color1, color2;
    float scale;
    
public:
    CheckerTexture(const Vector3& c1, const Vector3& c2, float scale = 10.0f);
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
};

// ==================== Image Texture ====================
class ImageTexture : public Texture {
private:
    unsigned char* data;
    int width, height, channels;
    
public:
    ImageTexture();
    ImageTexture(const char* filename);
    ~ImageTexture();
    
    bool load(const char* filename);
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
};

// ==================== Wood Texture ====================
class WoodTexture : public Texture {
private:
    float scale;
    Vector3 base_color;
    
public:
    WoodTexture(float scale = 5.0f, const Vector3& base_color = Vector3(0.6, 0.4, 0.2));
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
};

// ==================== Marble Texture ====================
class MarbleTexture : public Texture {
private:
    float scale;
    Vector3 base_color;
    
public:
    MarbleTexture(float scale = 3.0f, const Vector3& base_color = Vector3(0.8, 0.8, 0.9));
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
};

// ==================== Metal Texture ====================
class MetalTexture : public Texture {
private:
    float roughness_variation;
    
public:
    MetalTexture(float roughness_variation = 0.1f);
    virtual Vector3 value(double u, double v, const Vector3& p) const override;
    virtual float roughness_value(double u, double v, const Vector3& p) const override;
};

// ==================== Skybox Types ====================
enum class SkyboxType {
    SOLID = 0,
    GRADIENT = 1,
    SUNSET = 2,
    ATMOSPHERE = 3,
    IMAGE = 4
};

// ==================== Skybox ====================
class Skybox {
private:
    SkyboxType type;
    Vector3 colors[3];
    Vector3 horizon_color;
    Vector3 zenith_color;
    Vector3 ground_color;
    std::shared_ptr<ImageTexture> texture;
    
public:
    Skybox();
    
    void set_type(SkyboxType new_type) { type = new_type; }
    SkyboxType get_type() const { return type; }
    
    void set_colors(const Vector3& c1, const Vector3& c2 = Vector3(0,0,0), const Vector3& c3 = Vector3(0,0,0)) {
        colors[0] = c1;
        colors[1] = c2;
        colors[2] = c3;
    }
    
    void set_atmosphere_colors(const Vector3& horizon, const Vector3& zenith, const Vector3& ground) {
        horizon_color = horizon;
        zenith_color = zenith;
        ground_color = ground;
    }
    
    bool load_image(const char* filename);
    
    Vector3 get_color(const Vector3& direction) const;
};