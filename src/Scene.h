#pragma once
#include "Entity.h"
#include "Material.h"
#include "long_march.h"
#include <memory>
#include <vector>

struct InstanceInfo {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t has_normal;
    uint32_t normal_offset;
    uint32_t has_texcoord;
    uint32_t texcoord_offset;
    uint32_t has_tangent;
    uint32_t tangent_offset;
};

struct PointLight {
    glm::vec3 position;
    float power;
    glm::vec3 color;
    float radius;
};

struct AreaLight {
    glm::vec3 position;
    float power;
    glm::vec3 color;
    float _pad0;
    glm::vec3 u;
    float _pad1;
    glm::vec3 v;
    float _pad2;
};

struct SunLight {
    glm::vec3 direction;
    float power;
    glm::vec3 color;
    float angle; // in degrees
};

struct SceneInfo {
    uint32_t num_point_lights;
    uint32_t num_area_lights;
    uint32_t num_sun_lights;
    uint32_t has_hdri_skybox;
};

// Scene manages a collection of entities and builds the TLAS
class Scene {
  public:
    Scene(grassland::graphics::Core *core);
    ~Scene();

    // Add an entity to the scene
    void AddEntity(std::shared_ptr<Entity> entity);

    // Add a light to the scene
    void AddPointLight(const PointLight &light);

    // Add an area light to the scene
    void AddAreaLight(const AreaLight &light);

    // Add a sun light to the scene
    void AddSunLight(const SunLight &light);

    // Remove all entities
    void Clear();

    // Build/rebuild the TLAS from all entities
    void BuildAccelerationStructures();

    // Update TLAS instances (e.g., for animation)
    void UpdateInstances();

    void LoadEnvironmentMap(const std::string& filename);

    grassland::graphics::Image *GetEnvironmentMap() const {return environment_map_.get();}

    // Get the TLAS for rendering
    grassland::graphics::AccelerationStructure *GetTLAS() const { return tlas_.get(); }

    // Get materials buffer for all entities
    grassland::graphics::Buffer *GetMaterialsBuffer() const { return materials_buffer_.get(); }
    grassland::graphics::Buffer *GetGlobalVertexBuffer() const { return global_vertex_buffer_.get(); }
    grassland::graphics::Buffer *GetGlobalIndexBuffer() const { return global_index_buffer_.get(); }
    grassland::graphics::Buffer *GetGlobalNormalBuffer() const { return global_normal_buffer_.get(); }
    grassland::graphics::Buffer *GetGlobalTexcoordBuffer() const { return global_texcoord_buffer_.get(); }
    grassland::graphics::Buffer *GetGlobalTangentBuffer() const { return global_tangent_buffer_.get(); }
    grassland::graphics::Buffer *GetInstanceInfoBuffer() const { return instance_info_buffer_.get(); }
    grassland::graphics::Buffer *GetPointLightsBuffer() const { return point_lights_buffer_.get(); }
    grassland::graphics::Buffer *GetAreaLightsBuffer() const { return area_lights_buffer_.get(); }
    grassland::graphics::Buffer *GetSunLightsBuffer() const { return sun_lights_buffer_.get(); }
    grassland::graphics::Buffer *GetSceneInfoBuffer() const { return scene_info_buffer_.get(); }

    // Get all entities
    const std::vector<std::shared_ptr<Entity>> &GetEntities() const { return entities_; }

    // Get number of entities
    size_t GetEntityCount() const { return entities_.size(); }
    std::vector<grassland::graphics::Image *> GetMaterialImages() const;

  private:
    bool HasEnvironmentMap;
    void UpdateMaterialsBuffer();
    void UpdateGlobalBuffers();
    void UpdateLightsBuffer();

    grassland::graphics::Core *core_;
    std::vector<std::shared_ptr<Entity>> entities_;
    std::vector<PointLight> point_lights_;
    std::vector<AreaLight> area_lights_;
    std::vector<SunLight> sun_lights_;
    std::unique_ptr<grassland::graphics::AccelerationStructure> tlas_;
    std::unique_ptr<grassland::graphics::Buffer> global_vertex_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_index_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_normal_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_texcoord_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_tangent_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> instance_info_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> materials_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> point_lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> area_lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> sun_lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> scene_info_buffer_;
    std::vector<std::unique_ptr<grassland::graphics::Image>> material_images_;
    std::unique_ptr<grassland::graphics::Image> environment_map_;
};
