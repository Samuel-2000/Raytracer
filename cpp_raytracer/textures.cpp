// textures.cpp
#include "textures.h"
#include <cmath>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ==================== Noise Texture ====================
NoiseTexture::NoiseTexture(float scale) : scale(scale) {}

static double noise(double x, double y, double z) {
    // Simple pseudo-random noise (Perlin-like)
    auto fract = [](double val) { return val - floor(val); };
    auto hash = [](double n) {
        n = fract(n * 0.1031);
        n *= n + 33.33;
        n *= n + n;
        return fract(n);
    };
    
    int xi = (int)floor(x) & 255;
    int yi = (int)floor(y) & 255;
    int zi = (int)floor(z) & 255;
    
    double xf = x - floor(x);
    double yf = y - floor(y);
    double zf = z - floor(z);
    
    double u = xf * xf * (3.0 - 2.0 * xf);
    double v = yf * yf * (3.0 - 2.0 * yf);
    double w = zf * zf * (3.0 - 2.0 * zf);
    
    int a = hash(xi) + yi;
    int aa = hash(a) + zi;
    int ab = hash(a + 1) + zi;
    int b = hash(xi + 1) + yi;
    int ba = hash(b) + zi;
    int bb = hash(b + 1) + zi;
    
    double x1, x2, y1, y2;
    x1 = lerp(hash(aa), hash(ba), u);
    x2 = lerp(hash(ab), hash(bb), u);
    y1 = lerp(x1, x2, v);
    
    x1 = lerp(hash(aa + 1), hash(ba + 1), u);
    x2 = lerp(hash(ab + 1), hash(bb + 1), u);
    y2 = lerp(x1, x2, v);
    
    return lerp(y1, y2, w);
}

Vector3 NoiseTexture::value(double u, double v, const Vector3& p) const {
    double n = noise(scale * p.x, scale * p.y, scale * p.z);
    return Vector3(0.5, 0.5, 0.5) * (1.0 + n);
}

// ==================== Checker Texture ====================
CheckerTexture::CheckerTexture(const Vector3& c1, const Vector3& c2, float scale) 
    : color1(c1), color2(c2), scale(scale) {}

Vector3 CheckerTexture::value(double u, double v, const Vector3& p) const {
    double sines = sin(scale * p.x) * sin(scale * p.y) * sin(scale * p.z);
    if (sines < 0)
        return color1;
    else
        return color2;
}

// ==================== Image Texture ====================
ImageTexture::ImageTexture() : data(nullptr), width(0), height(0), channels(0) {}

ImageTexture::ImageTexture(const char* filename) : data(nullptr), width(0), height(0), channels(0) {
    load(filename);
}

ImageTexture::~ImageTexture() {
    if (data) stbi_image_free(data);
}

bool ImageTexture::load(const char* filename) {
    if (data) {
        stbi_image_free(data);
        data = nullptr;
    }
    data = stbi_load(filename, &width, &height, &channels, 0);
    return data != nullptr;
}

Vector3 ImageTexture::value(double u, double v, const Vector3& p) const {
    if (!data) return Vector3(0, 0, 0);
    
    // Clamp texture coordinates
    u = std::fmod(u, 1.0);
    if (u < 0) u += 1.0;
    v = 1.0 - std::fmod(v, 1.0); // Flip v for image coordinates
    if (v < 0) v += 1.0;
    
    int i = (int)(u * width);
    int j = (int)(v * height);
    
    i = std::max(0, std::min(width - 1, i));
    j = std::max(0, std::min(height - 1, j));
    
    unsigned char* pixel = data + (j * width + i) * channels;
    
    double r = pixel[0] / 255.0;
    double g = pixel[1] / 255.0;
    double b = pixel[2] / 255.0;
    
    return Vector3(r, g, b);
}

// ==================== Wood Texture ====================
WoodTexture::WoodTexture(float scale, const Vector3& base_color) 
    : scale(scale), base_color(base_color) {}

Vector3 WoodTexture::value(double u, double v, const Vector3& p) const {
    // Simple wood texture using noise
    double grain = noise(scale * p.x, scale * p.y, scale * p.z);
    grain = 0.5 + 0.5 * sin(100 * grain + 10 * p.y);
    return base_color * (0.7 + 0.3 * grain);
}

// ==================== Marble Texture ====================
MarbleTexture::MarbleTexture(float scale, const Vector3& base_color) 
    : scale(scale), base_color(base_color) {}

Vector3 MarbleTexture::value(double u, double v, const Vector3& p) const {
    double turbulence = 0.5 + 0.5 * noise(scale * p.x, scale * p.y, scale * p.z);
    double marble = 0.5 + 0.5 * sin(scale * p.x + 5 * turbulence);
    return base_color * (0.7 + 0.3 * marble);
}

// ==================== Metal Texture ====================
MetalTexture::MetalTexture(float roughness_variation) 
    : roughness_variation(roughness_variation) {}

Vector3 MetalTexture::value(double u, double v, const Vector3& p) const {
    // Metal texture with color variation
    double noise_val = noise(10 * p.x, 10 * p.y, 10 * p.z);
    return Vector3(0.7, 0.7, 0.7) * (0.9 + 0.1 * noise_val);
}

float MetalTexture::roughness_value(double u, double v, const Vector3& p) const {
    double noise_val = noise(10 * p.x, 10 * p.y, 10 * p.z);
    return roughness_variation * noise_val;
}

// ==================== Skybox ====================
Skybox::Skybox() : type(SkyboxType::GRADIENT) {
    colors[0] = Vector3(0.5, 0.7, 1.0); // Light blue
    colors[1] = Vector3(0.1, 0.2, 0.5); // Dark blue
    colors[2] = Vector3(0.0, 0.0, 0.0); // Not used in gradient
    horizon_color = Vector3(0.7, 0.8, 1.0);
    zenith_color = Vector3(0.1, 0.2, 0.5);
    ground_color = Vector3(0.1, 0.1, 0.1);
}

bool Skybox::load_image(const char* filename) {
    texture = std::make_shared<ImageTexture>(filename);
    return texture->load(filename);
}

Vector3 Skybox::get_color(const Vector3& direction) const {
    switch (type) {
        case SkyboxType::SOLID:
            return colors[0];
        case SkyboxType::GRADIENT: {
            // Simple gradient based on y coordinate
            double t = 0.5 * (direction.y + 1.0);
            return (1.0 - t) * colors[0] + t * colors[1];
        }
        case SkyboxType::SUNSET: {
            // Sunset colors
            double t = 0.5 * (direction.y + 1.0);
            Vector3 sunset_top(0.8, 0.3, 0.1);
            Vector3 sunset_bottom(0.1, 0.1, 0.3);
            return (1.0 - t) * sunset_bottom + t * sunset_top;
        }
        case SkyboxType::ATMOSPHERE: {
            // Atmosphere with horizon, zenith, and ground
            double y = direction.y;
            if (y > 0) {
                // Sky
                double t = y;
                return (1.0 - t) * horizon_color + t * zenith_color;
            } else {
                // Ground
                double t = -y;
                return (1.0 - t) * horizon_color + t * ground_color;
            }
        }
        case SkyboxType::IMAGE: {
            if (texture) {
                // Convert direction to spherical coordinates
                double phi = atan2(direction.z, direction.x);
                double theta = asin(direction.y);
                double u = 1.0 - (phi + M_PI) / (2.0 * M_PI);
                double v = (theta + M_PI/2) / M_PI;
                return texture->value(u, v, direction);
            } else {
                return Vector3(0,0,0);
            }
        }
        default:
            return Vector3(0,0,0);
    }
}