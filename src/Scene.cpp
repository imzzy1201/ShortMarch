#include "Scene.h"

Scene::Scene(grassland::graphics::Core *core) : core_(core) {}

Scene::~Scene() { Clear(); }

void Scene::AddEntity(std::shared_ptr<Entity> entity) {
    if (!entity || !entity->IsValid()) {
        grassland::LogError("Cannot add invalid entity to scene");
        return;
    }

    // Build BLAS for the entity
    entity->BuildBLAS(core_);

    entities_.push_back(entity);
    grassland::LogInfo("Added entity to scene (total: {})", entities_.size());
}

void Scene::Clear() {
    entities_.clear();
    point_lights_.clear();
    area_lights_.clear();
    sun_lights_.clear();
    tlas_.reset();
    materials_buffer_.reset();
    global_vertex_buffer_.reset();
    global_index_buffer_.reset();
    global_normal_buffer_.reset();
    global_texcoord_buffer_.reset();
    global_tangent_buffer_.reset();
    instance_info_buffer_.reset();
    point_lights_buffer_.reset();
    area_lights_buffer_.reset();
    sun_lights_buffer_.reset();
    scene_info_buffer_.reset();
    grassland::LogInfo("Cleared scene");
}

void Scene::AddPointLight(const PointLight &light) {
    point_lights_.push_back(light);
    UpdateLightsBuffer();
    grassland::LogInfo("Added point light to scene (total: {})", point_lights_.size());
}

void Scene::AddAreaLight(const AreaLight &light) {
    area_lights_.push_back(light);
    UpdateLightsBuffer();
    grassland::LogInfo("Added area light to scene (total: {})", area_lights_.size());
}

void Scene::AddSunLight(const SunLight &light) {
    sun_lights_.push_back(light);
    UpdateLightsBuffer();
    grassland::LogInfo("Added sun light to scene (total: {})", sun_lights_.size());
}

void Scene::LoadEnvironmentMap(const std::string& filename) {
    std::string full_path = grassland::FindAssetFile(filename);
    if (full_path.empty()) {
        grassland::LogError("Failed to find environment map file: {}", filename);
        return;
    }

    std::unique_ptr<grassland::graphics::Image> img;
    int res = grassland::graphics::LoadImageFromFile(core_, full_path, &img); 

    if (res == 0 && img) {
        environment_map_ = std::move(img);
        grassland::LogInfo("Loaded environment map: {}", full_path);
    } else {
        grassland::LogError("Failed to load environment map: {}", full_path);
    }
}

void Scene::BuildAccelerationStructures() {
    if (entities_.empty()) {
        grassland::LogWarning("No entities to build acceleration structures");
        return;
    }

    // Create TLAS instances from all entities
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto &entity = entities_[i];
        if (entity->GetBLAS()) {
            // Create instance with entity's transform
            // instanceCustomIndex is used to index into materials buffer
            // Convert mat4 to mat4x3 (drop the last row which is always [0,0,0,1] for affine transforms)
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());

            auto instance =
                entity->GetBLAS()->MakeInstance(transform_3x4,
                                                static_cast<uint32_t>(i), // instanceCustomIndex for material lookup
                                                0xFF,                     // instanceMask
                                                0,                        // instanceShaderBindingTableRecordOffset
                                                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE);
            instances.push_back(instance);
        }
    }

    // Build TLAS
    core_->CreateTopLevelAccelerationStructure(instances, &tlas_);
    grassland::LogInfo("Built TLAS with {} instances", instances.size());

    // Update materials buffer
    UpdateMaterialsBuffer();
    // Update global vertex/index/instance info buffers
    UpdateGlobalBuffers();
}

void Scene::UpdateInstances() {
    if (!tlas_ || entities_.empty()) {
        return;
    }

    // Recreate instances with updated transforms
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto &entity = entities_[i];
        if (entity->GetBLAS()) {
            // Convert mat4 to mat4x3
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());

            auto instance = entity->GetBLAS()->MakeInstance(transform_3x4, static_cast<uint32_t>(i), 0xFF, 0,
                                                            grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE);
            instances.push_back(instance);
        }
    }

    // Update TLAS
    tlas_->UpdateInstances(instances);
}

// void Scene::UpdateMaterialsBuffer() {
//     if (entities_.empty()) {
//         return;
//     }

//     // Collect all materials
//     std::vector<Material> materials;
//     materials.reserve(entities_.size());

//     for (const auto &entity : entities_) {
//         materials.push_back(entity->GetMaterial());
//     }

//     // Create/update materials buffer
//     size_t buffer_size = materials.size() * sizeof(Material);

//     if (!materials_buffer_) {
//         core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &materials_buffer_);
//     }

//     materials_buffer_->UploadData(materials.data(), buffer_size);
//     grassland::LogInfo("Updated materials buffer with {} materials", materials.size());
// }

std::vector<grassland::graphics::Image *> Scene::GetMaterialImages() const {
    std::vector<grassland::graphics::Image *> out;
    out.reserve(material_images_.size());
    for (const auto &u : material_images_) {
        out.push_back(u.get());
    }
    return out;
}

void Scene::UpdateMaterialsBuffer() {
    if (entities_.empty()) {
        return;
    }

    // Collect all materials
    struct GpuMaterial {
        glm::vec3 base_color;  // [DEPRECATED]

        glm::vec3 ambient;
        glm::vec3 diffuse;
        glm::vec3 specular;
        glm::vec3 transmittance;  // [NOT USED]
        glm::vec3 emission;
        float shininess;       // [NOT USED]
        float ior;             // index of refraction
        float dissolve;        // 1 == opaque; 0 == fully transparent
        int illum;             // illumination model, [NOT USED]

        int ambient_tex_id;             // [NOT USED]
        int diffuse_tex_id;             // [NOT USED]
        int specular_tex_id;            // [NOT USED]
        int specular_highlight_tex_id;  // [NOT USED]
        int bump_tex_id;                // [NOT USED]
        int displacement_tex_id;        // [NOT USED]
        int alpha_tex_id;               // [NOT USED]
        int reflection_tex_id;          // [NOT USED]

        // PBR extensions
        float roughness;    
        float metallic;  
        float sheen;                // [NOT USED]
        float clearcoat_thickness;  // [NOT USED]
        float clearcoat_roughness;  // [NOT USED]
        float anisotropy;           // [NOT USED]
        float anisotropy_rotation;  // [NOT USED]

        int roughness_tex_id;  // [NOT USED]
        int metallic_tex_id;   // [NOT USED]
        int sheen_tex_id;      // [NOT USED]
        int emissive_tex_id;   // [NOT USED]
        int normal_tex_id;     // [NOT USED]
    };



    std::vector<GpuMaterial> gpu_materials;
    gpu_materials.reserve(entities_.size());

    // Ensure we have a place to store material images; clear previous
    material_images_.clear();
    material_images_.reserve(entities_.size());

    for (const auto &entity : entities_) {
        const auto m = entity->GetMaterial();
        GpuMaterial gm{};
        gm.base_color = m.base_color;
        gm.ambient = m.ambient;
        gm.diffuse = m.diffuse;
        gm.specular = m.specular;
        gm.transmittance = m.transmittance;
        gm.emission = m.emission;
        gm.shininess = m.shininess;
        gm.ior = m.ior;
        gm.dissolve = m.dissolve;
        gm.illum = m.illum;
        gm.roughness = m.roughness;
        gm.metallic = m.metallic;

        gm.ambient_tex_id=m.ambient_tex_id;             // [NOT USED]
        gm.diffuse_tex_id = m.diffuse_tex_id;           // [NOT USED]
        gm.specular_tex_id = m.specular_tex_id;
        gm.specular_highlight_tex_id = m.specular_highlight_tex_id;            // [NOT USED]
        gm.bump_tex_id = m.bump_tex_id;  // [NOT USED]
        gm.displacement_tex_id = m.displacement_tex_id;                // [NOT USED]
        gm.alpha_tex_id = m.alpha_tex_id;        // [NOT USED]
        gm.reflection_tex_id = m.reflection_tex_id; 
                      // [NOT USED]
        gm.roughness = m.roughness;    
        gm.metallic = m.metallic;  
        gm.sheen = m.sheen;                // [NOT USED]
        gm.clearcoat_thickness = m.clearcoat_thickness;  // [NOT USED]
        gm.clearcoat_roughness = m.clearcoat_roughness;  // [NOT USED]
        gm.anisotropy = m.anisotropy;           // [NOT USED]
        gm.anisotropy_rotation = m.anisotropy_rotation;  // [NOT USED]

        gm.roughness_tex_id = m.roughness_tex_id;  // [NOT USED]
        gm.metallic_tex_id = m.metallic_tex_id;   // [NOT USED]
        gm.sheen_tex_id = m.sheen_tex_id;      // [NOT USED]
        gm.emissive_tex_id = m.emissive_tex_id;   // [NOT USED]
        gm.normal_tex_id = m.normal_tex_id;          // [NOT USED]

        if (!m.diffuse_color_texname.empty()) {
            std::string full_path = grassland::FindAssetFile(m.diffuse_color_texname);
            //grassland::LogInfo("test:{} {}", m.base_color_texname,full_path);
            if (!full_path.empty()) {
                std::unique_ptr<grassland::graphics::Image> img;
                int res = grassland::graphics::LoadImageFromFile(core_, full_path, &img);
                if (res == 0 && img) {
                    // store image and record its index
                    gm.diffuse_tex_id = static_cast<int>(material_images_.size());
                    grassland::LogInfo("add:{} {}", full_path,gm.diffuse_tex_id);
                    material_images_.push_back(std::move(img));
                }
            }
        }
        gpu_materials.push_back(gm);
    }

    // Create/update materials buffer (upload packed GPU materials)
    size_t buffer_size = gpu_materials.size() * sizeof(GpuMaterial);

    if (!materials_buffer_) {
        core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &materials_buffer_);
    }

    materials_buffer_->UploadData(gpu_materials.data(), buffer_size);
    grassland::LogInfo("Updated materials buffer with {} materials ({} images)", gpu_materials.size(), material_images_.size());
}

void Scene::UpdateGlobalBuffers() {
    if (entities_.empty()) {
        return;
    }

    std::vector<glm::vec3> all_vertices;
    std::vector<uint32_t> all_indices;
    std::vector<glm::vec3> all_normals;
    std::vector<glm::vec2> all_texcoords;
    std::vector<glm::vec3> all_tangents;
    std::vector<InstanceInfo> instance_infos;

    uint32_t current_vertex_offset = 0;
    uint32_t current_index_offset = 0;
    uint32_t current_normal_offset = 0;
    uint32_t current_texcoord_offset = 0;
    uint32_t current_tangent_offset = 0;

    for (const auto &entity : entities_) {
        const auto &mesh = entity->GetMesh();
        InstanceInfo info;
        info.vertex_offset = current_vertex_offset;
        info.index_offset = current_index_offset;
        info.has_normal = mesh.Normals() ? 1 : 0;
        info.normal_offset = current_normal_offset;
        info.has_texcoord = mesh.TexCoords() ? 1 : 0;
        info.texcoord_offset = current_texcoord_offset;
        info.has_tangent = mesh.Tangents() ? 1 : 0;
        info.tangent_offset = current_tangent_offset;
        instance_infos.push_back(info);

        // Append vertices (convert from Eigen::Vector3<float> to glm::vec3)
        const auto *eigen_positions = mesh.Positions();
        for (size_t vi = 0; vi < mesh.NumVertices(); ++vi) {
            const auto &p = eigen_positions[vi];
            all_vertices.emplace_back(p.x(), p.y(), p.z());
        }
        current_vertex_offset += mesh.NumVertices();

        // Append indices
        const uint32_t *indices = mesh.Indices();
        all_indices.insert(all_indices.end(), indices, indices + mesh.NumIndices());
        current_index_offset += mesh.NumIndices();

        if (info.has_normal) {
            // Append normals
            const auto *eigen_normals = mesh.Normals();
            for (size_t ni = 0; ni < mesh.NumVertices(); ++ni) {
                const auto &n = eigen_normals[ni];
                all_normals.emplace_back(n.x(), n.y(), n.z());
            }
            current_normal_offset += mesh.NumVertices();
        }

        if (info.has_texcoord) {
            // Append texcoords
            const auto *eigen_texcoords = mesh.TexCoords();
            for (size_t ti = 0; ti < mesh.NumVertices(); ++ti) {
                const auto &tc = eigen_texcoords[ti];
                all_texcoords.emplace_back(tc.x(), tc.y());
            }
            current_texcoord_offset += mesh.NumVertices();
        }

        if (info.has_tangent) {
            // Append tangents
            const auto *eigen_tangents = mesh.Tangents();
            for (size_t tai = 0; tai < mesh.NumVertices(); ++tai) {
                const auto &tan = eigen_tangents[tai];
                all_tangents.emplace_back(tan.x(), tan.y(), tan.z());
            }
            current_tangent_offset += mesh.NumVertices();
        }
    }

    // Create/Update buffers
    size_t vertex_buffer_size = all_vertices.size() * sizeof(glm::vec3);
    if (!global_vertex_buffer_ || global_vertex_buffer_->Size() < vertex_buffer_size) {
        core_->CreateBuffer(std::max(vertex_buffer_size, sizeof(glm::vec3)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_vertex_buffer_);
    }
    if (vertex_buffer_size > 0) {
        global_vertex_buffer_->UploadData(all_vertices.data(), vertex_buffer_size);
    }

    size_t index_buffer_size = all_indices.size() * sizeof(uint32_t);
    if (!global_index_buffer_ || global_index_buffer_->Size() < index_buffer_size) {
        core_->CreateBuffer(std::max(index_buffer_size, sizeof(uint32_t)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_index_buffer_);
    }
    if (index_buffer_size > 0) {
        global_index_buffer_->UploadData(all_indices.data(), index_buffer_size);
    }

    size_t normal_buffer_size = all_normals.size() * sizeof(glm::vec3);
    if (!global_normal_buffer_ || global_normal_buffer_->Size() < normal_buffer_size) {
        core_->CreateBuffer(std::max(normal_buffer_size, sizeof(glm::vec3)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_normal_buffer_);
    }
    if (normal_buffer_size > 0) {
        global_normal_buffer_->UploadData(all_normals.data(), normal_buffer_size);
    }

    size_t texcoord_buffer_size = all_texcoords.size() * sizeof(glm::vec2);
    if (!global_texcoord_buffer_ || global_texcoord_buffer_->Size() < texcoord_buffer_size) {
        core_->CreateBuffer(std::max(texcoord_buffer_size, sizeof(glm::vec2)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_texcoord_buffer_);
    }
    if (texcoord_buffer_size > 0) {
        global_texcoord_buffer_->UploadData(all_texcoords.data(), texcoord_buffer_size);
    }

    size_t tangent_buffer_size = all_tangents.size() * sizeof(glm::vec3);
    if (!global_tangent_buffer_ || global_tangent_buffer_->Size() < tangent_buffer_size) {
        core_->CreateBuffer(std::max(tangent_buffer_size, sizeof(glm::vec3)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_tangent_buffer_);
    }
    if (tangent_buffer_size > 0) {
        global_tangent_buffer_->UploadData(all_tangents.data(), tangent_buffer_size);
    }

    size_t instance_info_size = instance_infos.size() * sizeof(InstanceInfo);
    if (!instance_info_buffer_ || instance_info_buffer_->Size() < instance_info_size) {
        core_->CreateBuffer(std::max(instance_info_size, sizeof(InstanceInfo)),
                            grassland::graphics::BUFFER_TYPE_DYNAMIC, &instance_info_buffer_);
    }
    if (instance_info_size > 0) {
        instance_info_buffer_->UploadData(instance_infos.data(), instance_info_size);
    }

    grassland::LogInfo("Updated global buffers: {} vertices, {} indices", all_vertices.size(), all_indices.size());
}

void Scene::UpdateLightsBuffer() {
    // Update SceneInfo
    SceneInfo info;
    info.num_point_lights = static_cast<uint32_t>(point_lights_.size());
    info.num_area_lights = static_cast<uint32_t>(area_lights_.size());
    info.num_sun_lights = static_cast<uint32_t>(sun_lights_.size());
    info._pad = 0;

    if (!scene_info_buffer_) {
        core_->CreateBuffer(sizeof(SceneInfo), grassland::graphics::BUFFER_TYPE_DYNAMIC, &scene_info_buffer_);
    }
    scene_info_buffer_->UploadData(&info, sizeof(SceneInfo));

    // Update Point Lights
    size_t point_lights_size = point_lights_.size() * sizeof(PointLight);
    if (!point_lights_buffer_ || point_lights_buffer_->Size() < point_lights_size) {
        core_->CreateBuffer(std::max(point_lights_size, sizeof(PointLight)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &point_lights_buffer_);
    }
    if (point_lights_size > 0) {
        point_lights_buffer_->UploadData(point_lights_.data(), point_lights_size);
    }

    // Update Area Lights
    size_t area_lights_size = area_lights_.size() * sizeof(AreaLight);
    if (!area_lights_buffer_ || area_lights_buffer_->Size() < area_lights_size) {
        core_->CreateBuffer(std::max(area_lights_size, sizeof(AreaLight)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &area_lights_buffer_);
    }
    if (area_lights_size > 0) {
        area_lights_buffer_->UploadData(area_lights_.data(), area_lights_size);
    }

    // Update Sun Lights
    size_t sun_lights_size = sun_lights_.size() * sizeof(SunLight);
    if (!sun_lights_buffer_ || sun_lights_buffer_->Size() < sun_lights_size) {
        core_->CreateBuffer(std::max(sun_lights_size, sizeof(SunLight)), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &sun_lights_buffer_);
    }
    if (sun_lights_size > 0) {
        sun_lights_buffer_->UploadData(sun_lights_.data(), sun_lights_size);
    }

    grassland::LogInfo("Updated lights buffer: {} point lights, {} area lights, {} sun lights", point_lights_.size(),
                       area_lights_.size(), sun_lights_.size());
}