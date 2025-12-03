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
    instance_info_buffer_.reset();
    lights_buffer_.reset();
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

void Scene::UpdateMaterialsBuffer() {
    if (entities_.empty()) {
        return;
    }

    // Collect all materials
    std::vector<Material> materials;
    materials.reserve(entities_.size());

    for (const auto &entity : entities_) {
        materials.push_back(entity->GetMaterial());
    }

    // Create/update materials buffer
    size_t buffer_size = materials.size() * sizeof(Material);

    if (!materials_buffer_) {
        core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &materials_buffer_);
    }

    materials_buffer_->UploadData(materials.data(), buffer_size);
    grassland::LogInfo("Updated materials buffer with {} materials", materials.size());
}

void Scene::UpdateGlobalBuffers() {
    if (entities_.empty()) {
        return;
    }

    std::vector<glm::vec3> all_vertices;
    std::vector<uint32_t> all_indices;
    std::vector<InstanceInfo> instance_infos;

    uint32_t current_vertex_offset = 0;
    uint32_t current_index_offset = 0;

    for (const auto &entity : entities_) {
        const auto &mesh = entity->GetMesh();

        // Append vertices (convert from Eigen::Vector3<float> to glm::vec3)
        const auto *eigen_positions = mesh.Positions();
        for (size_t vi = 0; vi < mesh.NumVertices(); ++vi) {
            const auto &p = eigen_positions[vi];
            all_vertices.emplace_back(p.x(), p.y(), p.z());
        }

        // Append indices
        const uint32_t *indices = mesh.Indices();
        all_indices.insert(all_indices.end(), indices, indices + mesh.NumIndices());

        // Record offsets
        InstanceInfo info;
        info.vertex_offset = current_vertex_offset;
        info.index_offset = current_index_offset;
        instance_infos.push_back(info);

        current_vertex_offset += mesh.NumVertices();
        current_index_offset += mesh.NumIndices();
    }

    // Create/Update buffers
    if (!global_vertex_buffer_ || global_vertex_buffer_->Size() < all_vertices.size() * sizeof(glm::vec3)) {
        core_->CreateBuffer(all_vertices.size() * sizeof(glm::vec3), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_vertex_buffer_);
    }
    global_vertex_buffer_->UploadData(all_vertices.data(), all_vertices.size() * sizeof(glm::vec3));

    if (!global_index_buffer_ || global_index_buffer_->Size() < all_indices.size() * sizeof(uint32_t)) {
        core_->CreateBuffer(all_indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &global_index_buffer_);
    }
    global_index_buffer_->UploadData(all_indices.data(), all_indices.size() * sizeof(uint32_t));

    if (!instance_info_buffer_ || instance_info_buffer_->Size() < instance_infos.size() * sizeof(InstanceInfo)) {
        core_->CreateBuffer(instance_infos.size() * sizeof(InstanceInfo), grassland::graphics::BUFFER_TYPE_DYNAMIC,
                            &instance_info_buffer_);
    }
    instance_info_buffer_->UploadData(instance_infos.data(), instance_infos.size() * sizeof(InstanceInfo));

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
    if (!point_lights_.empty()) {
        size_t buffer_size = point_lights_.size() * sizeof(PointLight);
        if (!lights_buffer_ || lights_buffer_->Size() < buffer_size) {
            core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &lights_buffer_);
        }
        lights_buffer_->UploadData(point_lights_.data(), buffer_size);
    }

    // Update Area Lights
    if (!area_lights_.empty()) {
        size_t buffer_size = area_lights_.size() * sizeof(AreaLight);
        if (!area_lights_buffer_ || area_lights_buffer_->Size() < buffer_size) {
            core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &area_lights_buffer_);
        }
        area_lights_buffer_->UploadData(area_lights_.data(), buffer_size);
    }

    // Update Sun Lights
    if (!sun_lights_.empty()) {
        size_t buffer_size = sun_lights_.size() * sizeof(SunLight);
        if (!sun_lights_buffer_ || sun_lights_buffer_->Size() < buffer_size) {
            core_->CreateBuffer(buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &sun_lights_buffer_);
        }
        sun_lights_buffer_->UploadData(sun_lights_.data(), buffer_size);
    }

    grassland::LogInfo("Updated lights buffer: {} point lights, {} area lights, {} sun lights", point_lights_.size(),
                       area_lights_.size(), sun_lights_.size());
}