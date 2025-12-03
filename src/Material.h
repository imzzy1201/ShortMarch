#pragma once
#include "long_march.h"
#include "tiny_obj_loader.h"

// Simple material structure for ray tracing
struct Material {
    glm::vec3 base_color;            // DEPRECATED
    float roughness;                 // DEPRECATED
    float metallic;                  // DEPRECATED
    tinyobj::material_t tinyobj_mat; // TinyObjLoader material

    Material() : base_color(0.8f, 0.8f, 0.8f), roughness(0.5f), metallic(0.0f) {}

    Material(const glm::vec3 &color, float rough = 0.5f, float metal = 0.0f)
        : base_color(color), roughness(rough), metallic(metal) {}

    Material(const tinyobj::material_t &mat)
        : base_color(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]), roughness(mat.roughness), metallic(mat.metallic),
          tinyobj_mat(mat) {}
};
