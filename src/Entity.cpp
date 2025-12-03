#include "Entity.h"
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

Entity::Entity(const std::string &obj_file_path, const Material &material, const glm::mat4 &transform)
    : material_(material), transform_(transform), mesh_loaded_(false) {

    LoadMesh(obj_file_path);
}

Entity::Entity(grassland::Mesh<float> &&mesh, const Material &material, const glm::mat4 &transform)
    : mesh_(std::move(mesh)), material_(material), transform_(transform), mesh_loaded_(true) {}

Entity::~Entity() {
    blas_.reset();
    index_buffer_.reset();
    vertex_buffer_.reset();
}

bool Entity::LoadMesh(const std::string &obj_file_path) {
    // Try to load the OBJ file
    std::string full_path = grassland::FindAssetFile(obj_file_path);

    if (mesh_.LoadObjFile(full_path) != 0) {
        grassland::LogError("Failed to load mesh from: {}", obj_file_path);
        mesh_loaded_ = false;
        return false;
    }

    grassland::LogInfo("Successfully loaded mesh: {} ({} vertices, {} indices)", obj_file_path, mesh_.NumVertices(),
                       mesh_.NumIndices());

    mesh_loaded_ = true;
    return true;
}

std::vector<std::unique_ptr<Entity>> Entity::LoadEntitiesFromObjWithMaterials(const std::string &obj_file_path,
                                                                              const glm::mat4 &transform) {
    std::vector<std::unique_ptr<Entity>> result;

    std::string full_path = grassland::FindAssetFile(obj_file_path);
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(full_path).parent_path().string();
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(full_path, reader_config)) {
        if (!reader.Error().empty()) {
            grassland::LogError("TinyObjReader: {}", reader.Error());
        }
        return result;
    }

    if (!reader.Warning().empty()) {
        grassland::LogWarning("TinyObjReader: {}", reader.Warning());
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    struct Key {
        int vi, ni, ti;
    };

    struct PerMat {
        std::vector<grassland::Vector3<float>> positions;
        std::vector<grassland::Vector3<float>> normals;
        std::vector<grassland::Vector2<float>> texcoords;
        std::vector<uint32_t> indices;
        std::map<std::tuple<int, int, int>, uint32_t> index_map;
    };

    std::map<int, PerMat> per_material_data;

    size_t global_index_offset = 0;
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int material_id = shapes[s].mesh.material_ids[f];
            auto &pm = per_material_data[material_id];

            size_t fv = shapes[s].mesh.num_face_vertices[f];
            // for each vertex in face
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                int vi = idx.vertex_index;
                int ni = idx.normal_index;
                int ti = idx.texcoord_index;

                std::tuple<int, int, int> key = std::make_tuple(vi, ni, ti);

                auto it = pm.index_map.find(key);
                uint32_t local_index;
                if (it == pm.index_map.end()) {
                    // create new local vertex
                    tinyobj::real_t vx = attrib.vertices[3 * (size_t)vi + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * (size_t)vi + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * (size_t)vi + 2];
                    pm.positions.emplace_back((float)vx, (float)vy, (float)vz);

                    if (ni >= 0) {
                        tinyobj::real_t nx = attrib.normals[3 * (size_t)ni + 0];
                        tinyobj::real_t ny = attrib.normals[3 * (size_t)ni + 1];
                        tinyobj::real_t nz = attrib.normals[3 * (size_t)ni + 2];
                        pm.normals.emplace_back((float)nx, (float)ny, (float)nz);
                    }
                    if (ti >= 0) {
                        tinyobj::real_t tx = attrib.texcoords[2 * (size_t)ti + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * (size_t)ti + 1];
                        pm.texcoords.emplace_back((float)tx, (float)ty);
                    }

                    local_index = static_cast<uint32_t>(pm.positions.size() - 1);
                    pm.index_map.emplace(key, local_index);
                } else {
                    local_index = it->second;
                }

                if (v >= 2) {
                    tinyobj::index_t idx0 = shapes[s].mesh.indices[index_offset + 0];
                    tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset + v - 1];
                    tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + v];

                    std::tuple<int, int, int> key0 =
                        std::make_tuple(idx0.vertex_index, idx0.normal_index, idx0.texcoord_index);
                    std::tuple<int, int, int> key1 =
                        std::make_tuple(idx1.vertex_index, idx1.normal_index, idx1.texcoord_index);
                    std::tuple<int, int, int> key2 =
                        std::make_tuple(idx2.vertex_index, idx2.normal_index, idx2.texcoord_index);

                    uint32_t li0 = pm.index_map[key0];
                    uint32_t li1 = pm.index_map[key1];
                    uint32_t li2 = pm.index_map[key2];

                    pm.indices.push_back(li0);
                    pm.indices.push_back(li1);
                    pm.indices.push_back(li2);
                }
            }

            index_offset += fv;
            global_index_offset += fv;
        }
    }

    for (auto &kv : per_material_data) {
        int mat_id = kv.first;
        PerMat &pm = kv.second;

        if (pm.positions.empty() || pm.indices.empty())
            continue;

        grassland::Mesh<float> m(pm.positions.size(), pm.indices.size(), pm.indices.data(), pm.positions.data(),
                                 pm.normals.empty() ? nullptr : pm.normals.data(),
                                 pm.texcoords.empty() ? nullptr : pm.texcoords.data(), nullptr);

        Material mat;
        if (mat_id >= 0 && mat_id < (int)materials.size()) {
            mat = Material(materials[mat_id]);
        } else {
            mat = Material();
        }

        auto ent = std::make_unique<Entity>(std::move(m), mat, transform);
        result.push_back(std::move(ent));
    }

    grassland::LogInfo("Loaded OBJ '{}' into {} entities (by material)", obj_file_path, result.size());

    return result;
}

void Entity::BuildBLAS(grassland::graphics::Core *core) {
    if (!mesh_loaded_) {
        grassland::LogError("Cannot build BLAS: mesh not loaded");
        return;
    }

    // Create vertex buffer
    size_t vertex_buffer_size = mesh_.NumVertices() * sizeof(glm::vec3);
    core->CreateBuffer(vertex_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
    vertex_buffer_->UploadData(mesh_.Positions(), vertex_buffer_size);

    // Create index buffer
    size_t index_buffer_size = mesh_.NumIndices() * sizeof(uint32_t);
    core->CreateBuffer(index_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
    index_buffer_->UploadData(mesh_.Indices(), index_buffer_size);

    // Build BLAS
    core->CreateBottomLevelAccelerationStructure(vertex_buffer_.get(), index_buffer_.get(), sizeof(glm::vec3), &blas_);

    grassland::LogInfo("Built BLAS for entity");
}
