// vector3.h
#pragma once
#include <cmath>  // Add this line

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
        double inv_scalar = 1.0 / scalar;
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

    Vector3 copy() const {
        return Vector3(x, y, z);
    }

};

inline Vector3 operator*(double scalar, const Vector3& vec) {
    return vec * scalar;
}

inline Vector3 lerp(const Vector3& a, const Vector3& b, double t) {
    return a * (1.0 - t) + b * t;
}