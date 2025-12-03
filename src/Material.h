#pragma once
#include "long_march.h"
#include "tiny_obj_loader.h"

// Simple material structure for ray tracing
struct Material {
    glm::vec3 base_color = glm::vec3(0.8f, 0.8f, 0.8f); // DEPRECATED
    float roughness = 0.5f;                             // DEPRECATED
    float metallic = 0.0f;                              // DEPRECATED

    glm::vec3 ambient = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 diffuse = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 specular = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 transmittance = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 emission = glm::vec3(0.0f, 0.0f, 0.0f);
    float shininess = 1.0f;
    float ior = 1.0f;      // index of refraction
    float dissolve = 1.0f; // 1 == opaque; 0 == fully transparent
    int illum = 0;         // illumination model

    int ambient_tex_id = -1;
    int diffuse_tex_id = -1;
    int specular_tex_id = -1;
    int specular_highlight_tex_id = -1;
    int bump_tex_id = -1;
    int displacement_tex_id = -1;
    int alpha_tex_id = -1;
    int reflection_tex_id = -1;

    // tinyobj::material_t tinyobj_mat; // TinyObjLoader material

    Material() : base_color(0.8f, 0.8f, 0.8f), roughness(0.5f), metallic(0.0f) {}

    Material(const glm::vec3 &color, float rough = 0.5f, float metal = 0.0f)
        : base_color(color), roughness(rough), metallic(metal) {}

    Material(const tinyobj::material_t &mat)
        : base_color(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]), roughness(mat.roughness), metallic(mat.metallic) {
        ambient = glm::vec3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
        diffuse = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        specular = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
        transmittance = glm::vec3(mat.transmittance[0], mat.transmittance[1], mat.transmittance[2]);
        emission = glm::vec3(mat.emission[0], mat.emission[1], mat.emission[2]);
        shininess = mat.shininess;
        ior = mat.ior;
        dissolve = mat.dissolve;
        illum = mat.illum;
    }
};
