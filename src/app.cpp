#include "app.h"
#include "Material.h"
#include "Entity.h"

#include "glm/gtc/matrix_transform.hpp"
#include "imgui.h"

namespace {
#include "built_in_shaders.inl"
}

Application::Application(grassland::graphics::BackendAPI api) {
    grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
    core_->InitializeLogicalDeviceAutoSelect(true);

    grassland::LogInfo("Device Name: {}", core_->DeviceName());
    grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
    core_.reset();
}

// Event handler for keyboard input
// Poll keyboard state directly to ensure it works even when ImGui is active
void Application::ProcessInput() {
    // Only process input if camera is enabled
    if (!camera_enabled_) {
        return;
    }

    // Get GLFW window handle
    GLFWwindow* glfw_window = window_->GLFWWindow();
    
    // Check if this window has focus - only process input for focused window
    if (glfwGetWindowAttrib(glfw_window, GLFW_FOCUSED) == GLFW_FALSE) {
        return;
    }

    // Poll key states directly
    // Move forward
    if (glfwGetKey(glfw_window, GLFW_KEY_W) == GLFW_PRESS) {
        camera_pos_ += camera_speed_ * camera_front_;
    }
    // Move backward
    if (glfwGetKey(glfw_window, GLFW_KEY_S) == GLFW_PRESS) {
        camera_pos_ -= camera_speed_ * camera_front_;
    }
    // Strafe left
    if (glfwGetKey(glfw_window, GLFW_KEY_A) == GLFW_PRESS) {
        camera_pos_ -= glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed_;
    }
    // Strafe right
    if (glfwGetKey(glfw_window, GLFW_KEY_D) == GLFW_PRESS) {
        camera_pos_ += glm::normalize(glm::cross(camera_front_, camera_up_)) * camera_speed_;
    }
    // Move up (Space)
    if (glfwGetKey(glfw_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        camera_pos_ += camera_speed_ * camera_up_;
    }
    // Move down (Shift)
    if (glfwGetKey(glfw_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
        glfwGetKey(glfw_window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        camera_pos_ -= camera_speed_ * camera_up_;
    }
}

// Event handler for mouse movement
void Application::OnMouseMove(double xpos, double ypos) {
    // Always store mouse position for hover detection (even if ImGui wants input)
    mouse_x_ = xpos;
    mouse_y_ = ypos;

    // Only process camera look if camera is enabled
    if (!camera_enabled_) {
        return;
    }

    if (first_mouse_) {
        last_x_ = (float)xpos;
        last_y_ = (float)ypos;
        first_mouse_ = false;
        return;
    }

    float xoffset = (float)xpos - last_x_;
    float yoffset = last_y_ - (float)ypos; // Reversed since y-coordinates go from bottom to top
    last_x_ = (float)xpos;
    last_y_ = (float)ypos;

    xoffset *= mouse_sensitivity_;
    yoffset *= mouse_sensitivity_;

    yaw_ += xoffset;
    pitch_ += yoffset;

    // Constrain pitch to avoid flipping
    if (pitch_ > 89.0f)
        pitch_ = 89.0f;
    if (pitch_ < -89.0f)
        pitch_ = -89.0f;

    // Recalculate the camera_front_ vector
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    camera_front_ = glm::normalize(front);
}

// Event handler for mouse button clicks
void Application::OnMouseButton(int button, int action, int mods, double xpos, double ypos) {
    const int BUTTON_LEFT = 0;  // Left mouse button
    const int BUTTON_RIGHT = 1; // Right mouse button
    const int ACTION_PRESS = 1;

    // Left-click to select entity (only when camera is disabled)
    if (button == BUTTON_LEFT && action == ACTION_PRESS && !camera_enabled_) {
        // Select the currently hovered entity
        if (hovered_entity_id_ >= 0) {
            selected_entity_id_ = hovered_entity_id_;
            grassland::LogInfo("Selected Entity #{}", selected_entity_id_);
        } else {
            selected_entity_id_ = -1;
            grassland::LogInfo("Deselected entity");
        }
    }

    if (button == BUTTON_RIGHT && action == ACTION_PRESS) {
        // Toggle camera mode
        camera_enabled_ = !camera_enabled_;
        
        GLFWwindow* glfw_window = window_->GLFWWindow();
        
        if (camera_enabled_) {
            // Entering camera mode - hide cursor and grab it
            glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            first_mouse_ = true; // Reset to prevent jump
            grassland::LogInfo("Camera mode enabled - use WASD/Space/Shift to move, mouse to look");
        } else {
            // Exiting camera mode - show cursor
            glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            grassland::LogInfo("Camera mode disabled - cursor visible, showing info overlay");
        }
    }
}



void Application::OnInit() {
    alive_ = true;
    core_->CreateWindowObject(1280, 720,
        ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") +
        std::string(" Ray Tracing Scene Demo"),
        &window_);

    // Initialize ImGui for this window
    window_->InitImGui();

    // Register the mouse move event handler
    window_->MouseMoveEvent().RegisterCallback(
        [this](double xpos, double ypos) {
            this->OnMouseMove(xpos, ypos);
        }
    );
    // Register the mouse button event handler
    window_->MouseButtonEvent().RegisterCallback(
        [this](int button, int action, int mods, double xpos, double ypos) {
            this->OnMouseButton(button, action, mods, xpos, ypos);
        }
    );

    // Initialize camera as DISABLED to avoid cursor conflicts with multiple windows
    camera_enabled_ = false;
    hovered_entity_id_ = -1; // No entity hovered initially
    selected_entity_id_ = -1; // No entity selected initially
    mouse_x_ = 0.0;
    mouse_y_ = 0.0;
    // Don't grab cursor initially - user can right-click to enable camera mode

    // Create scene
    scene_ = std::make_unique<Scene>(core_.get());

    // Add entities to the scene
    // Ground plane - a cube scaled to be flat
    {
        auto ground = std::make_shared<Entity>(
            "meshes/cube.obj",
            Material(glm::vec3(0.8f, 0.8f, 0.8f), 0.8f, 0.0f),
            glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, 0.0f)), 
                      glm::vec3(10.0f, 0.1f, 10.0f))
        );
        scene_->AddEntity(ground);
    }

    // Red sphere (using octahedron as sphere substitute)
    {
        auto red_sphere = std::make_shared<Entity>(
            "meshes/octahedron.obj",
            Material(glm::vec3(1.0f, 0.2f, 0.2f), 0.3f, 0.0f),
            glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(red_sphere);
    }

    // Green metallic sphere
    {
        auto green_sphere = std::make_shared<Entity>(
            "meshes/octahedron.obj",
            Material(glm::vec3(0.2f, 1.0f, 0.2f), 0.2f, 0.8f),
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(green_sphere);
    }

    // Blue cube
    {
        auto blue_cube = std::make_shared<Entity>(
            "meshes/cube.obj",
            Material(glm::vec3(0.2f, 0.2f, 1.0f), 0.5f, 0.0f),
            glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.5f, 0.0f))
        );
        scene_->AddEntity(blue_cube);
    }

    // Build acceleration structures
    scene_->BuildAccelerationStructures();

    core_->CreateBuffer(sizeof(CameraObject), grassland::graphics::BUFFER_TYPE_DYNAMIC, &camera_object_buffer_);
    
    // Create hover info buffer
    core_->CreateBuffer(sizeof(HoverInfo), grassland::graphics::BUFFER_TYPE_DYNAMIC, &hover_info_buffer_);
    HoverInfo initial_hover{};
    initial_hover.hovered_entity_id = -1;
    hover_info_buffer_->UploadData(&initial_hover, sizeof(HoverInfo));

    // Initialize camera state member variables
    camera_pos_ = glm::vec3{ 0.0f, 1.0f, 5.0f };
    camera_up_ = glm::vec3{ 0.0f, 1.0f, 0.0f }; // World up
    camera_speed_ = 0.01f;

    // Initialize new mouse/view variables
    yaw_ = -90.0f; // Point down -Z
    pitch_ = 0.0f;
    last_x_ = (float)window_->GetWidth() / 2.0f;
    last_y_ = (float)window_->GetHeight() / 2.0f;
    mouse_sensitivity_ = 0.1f;
    first_mouse_ = true;

    // Calculate initial camera_front_ based on yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front.y = sin(glm::radians(pitch_));
    front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    camera_front_ = glm::normalize(front);

    // Set initial camera buffer data
    CameraObject camera_object{};
    camera_object.screen_to_camera = glm::inverse(
        glm::perspective(glm::radians(60.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
    camera_object.camera_to_world =
        glm::inverse(glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_));
    camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));

    core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
        &color_image_);

    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "RayGenMain", "lib_6_3", &raygen_shader_);
    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "MissMain", "lib_6_3", &miss_shader_);
    core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "ClosestHitMain", "lib_6_3", &closest_hit_shader_);
    grassland::LogInfo("Shader compiled successfully");

    core_->CreateRayTracingProgram(raygen_shader_.get(), miss_shader_.get(), closest_hit_shader_.get(), &program_);
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);  // space0
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);          // space1
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space2
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);          // space3 - materials
    program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);          // space4 - hover info
    program_->Finalize();
}

void Application::OnClose() {
    // Clean up graphics resources first
    program_.reset();
    raygen_shader_.reset();
    miss_shader_.reset();
    closest_hit_shader_.reset();

    scene_.reset();

    color_image_.reset();
    camera_object_buffer_.reset();
    hover_info_buffer_.reset();
    
    // Don't call TerminateImGui - let the window destructor handle it
    // Just reset window which will clean everything up properly
    window_.reset();
}

void Application::UpdateHoveredEntity() {
    // Only detect hover when camera is disabled (cursor visible)
    if (camera_enabled_) {
        hovered_entity_id_ = -1;
        return;
    }

    // Cast ray from mouse position
    float x = (float)mouse_x_;
    float y = (float)mouse_y_;
    float width = (float)window_->GetWidth();
    float height = (float)window_->GetHeight();
    
    // Mouse position is tracked continuously, no logging needed

    // Match the shader's ray generation exactly (shader.hlsl lines 32-38)
    // Note: mouse_x_ and mouse_y_ are already pixel coordinates [0, width) x [0, height)
    // In shader: pixel_center = DispatchRaysIndex() + (0.5, 0.5)
    // But mouse coords are already continuous, so we don't add 0.5
    
    // 1. uv = mouse / dimensions -> [0, 1]
    glm::vec2 uv = glm::vec2(x / width, y / height);
    
    // 2. Flip Y (shader does: uv.y = 1.0 - uv.y)
    uv.y = 1.0f - uv.y;
    
    // 3. Convert to NDC: d = uv * 2.0 - 1.0 -> [-1, 1]
    glm::vec2 d = uv * 2.0f - 1.0f;

    // Get the camera matrices - must match shader exactly
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), width / height, 0.1f, 10.0f);
    glm::mat4 view = glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_);
    
    glm::mat4 screen_to_camera = glm::inverse(projection);
    glm::mat4 camera_to_world = glm::inverse(view);

    // 4. target = screen_to_camera * (d, 1, 1)
    glm::vec4 target = screen_to_camera * glm::vec4(d.x, d.y, 1.0f, 1.0f);
    
    // 5. direction = camera_to_world * (target.xyz, 0)
    glm::vec4 direction = camera_to_world * glm::vec4(glm::vec3(target), 0.0f);

    // Ray origin and direction
    glm::vec3 ray_origin = camera_pos_;
    glm::vec3 ray_world = glm::normalize(glm::vec3(direction));

    // Simple ray-sphere intersection for each entity
    hovered_entity_id_ = -1;
    float closest_t = FLT_MAX;

    const auto& entities = scene_->GetEntities();
    for (size_t i = 0; i < entities.size(); i++) {
        const auto& entity = entities[i];
        
        // Extract position from entity transform (assumes translation in last column)
        glm::mat4 transform = entity->GetTransform();
        glm::vec3 center = glm::vec3(transform[3]);
        
        // Approximate radius based on transform scale
        // Use a generous radius multiplier to make selection easier
        glm::vec3 scale = glm::vec3(
            glm::length(glm::vec3(transform[0])),
            glm::length(glm::vec3(transform[1])),
            glm::length(glm::vec3(transform[2]))
        );
        float radius = glm::max(glm::max(scale.x, scale.y), scale.z) * 1.5f;

        // Ray-sphere intersection
        glm::vec3 oc = ray_origin - center;
        float a = glm::dot(ray_world, ray_world);
        float b = 2.0f * glm::dot(oc, ray_world);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant >= 0) {
            float t = (-b - sqrt(discriminant)) / (2.0f * a);
            if (t > 0 && t < closest_t) {
                closest_t = t;
                hovered_entity_id_ = static_cast<int>(i);
            }
        }
    }
    
    // Hover state is shown in the UI panels, no logging needed
}

void Application::OnUpdate() {
    if (window_->ShouldClose()) {
        window_->CloseWindow();
        alive_ = false;
        return;  // Exit update immediately after closing
    }
    if (alive_) {
        // Process keyboard input to move camera
        ProcessInput();
        
        // Update which entity is being hovered
        UpdateHoveredEntity();
        
        // Update hover info buffer
        HoverInfo hover_info{};
        hover_info.hovered_entity_id = hovered_entity_id_;
        hover_info_buffer_->UploadData(&hover_info, sizeof(HoverInfo));

        // Update the camera buffer with new position/orientation
        CameraObject camera_object{};
        camera_object.screen_to_camera = glm::inverse(
            glm::perspective(glm::radians(60.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
        camera_object.camera_to_world =
            glm::inverse(glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_));
        camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));


        // Optional: Animate entities
        // For now, entities are static. You can update their transforms and call:
        // scene_->UpdateInstances();
    }
}

void Application::RenderInfoOverlay() {
    // Only show overlay when camera is disabled
    if (camera_enabled_) {
        return;
    }

    // Create a window on the left side (matching entity panel style)
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(350.0f, (float)window_->GetHeight()), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                     ImGuiWindowFlags_NoResize | 
                                     ImGuiWindowFlags_NoCollapse;
    
    if (!ImGui::Begin("Scene Information", nullptr, window_flags)) {
        ImGui::End();
        return;
    }

    // Camera Information
    ImGui::SeparatorText("Camera");
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", camera_pos_.x, camera_pos_.y, camera_pos_.z);
    ImGui::Text("Direction: (%.2f, %.2f, %.2f)", camera_front_.x, camera_front_.y, camera_front_.z);
    ImGui::Text("Yaw: %.1f°  Pitch: %.1f°", yaw_, pitch_);
    ImGui::Text("Speed: %.3f", camera_speed_);
    ImGui::Text("Sensitivity: %.2f", mouse_sensitivity_);

    ImGui::Spacing();

    // Scene Information
    ImGui::SeparatorText("Scene");
    size_t entity_count = scene_->GetEntityCount();
    ImGui::Text("Entities: %zu", entity_count);
    ImGui::Text("Materials: %zu", entity_count); // One material per entity
    
    // Show hovered entity
    if (hovered_entity_id_ >= 0) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Hovered: Entity #%d", hovered_entity_id_);
    } else {
        ImGui::Text("Hovered: None");
    }
    
    // Show selected entity
    if (selected_entity_id_ >= 0) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Selected: Entity #%d", selected_entity_id_);
    } else {
        ImGui::Text("Selected: None");
    }
    
    // Calculate total triangles
    size_t total_triangles = 0;
    for (const auto& entity : scene_->GetEntities()) {
        if (entity && entity->GetIndexBuffer()) {
            // Each 3 indices = 1 triangle
            size_t indices = entity->GetIndexBuffer()->Size() / sizeof(uint32_t);
            total_triangles += indices / 3;
        }
    }
    ImGui::Text("Total Triangles: %zu", total_triangles);

    ImGui::Spacing();

    // Render Information
    ImGui::SeparatorText("Render");
    ImGui::Text("Resolution: %d x %d", window_->GetWidth(), window_->GetHeight());
    ImGui::Text("Backend: %s", 
                core_->API() == grassland::graphics::BACKEND_API_VULKAN ? "Vulkan" : "D3D12");
    ImGui::Text("Device: %s", core_->DeviceName().c_str());

    ImGui::Spacing();

    // Controls hint
    ImGui::SeparatorText("Controls");
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Right Click to enable camera");
    ImGui::Text("W/A/S/D - Move camera");
    ImGui::Text("Space/Shift - Up/Down");
    ImGui::Text("Mouse - Look around");

    ImGui::End();
}

void Application::RenderEntityPanel() {
    // Only show entity panel when camera is disabled
    if (camera_enabled_) {
        return;
    }

    // Create a window on the right side
    ImGui::SetNextWindowPos(ImVec2((float)window_->GetWidth() - 350.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(350.0f, (float)window_->GetHeight()), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                     ImGuiWindowFlags_NoResize | 
                                     ImGuiWindowFlags_NoCollapse;
    
    if (!ImGui::Begin("Entity Inspector", nullptr, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::SeparatorText("Entity Selection");
    
    const auto& entities = scene_->GetEntities();
    size_t entity_count = entities.size();
    
    // Entity dropdown with limited height
    ImGui::Text("Select Entity:");
    
    // Create preview text
    std::string preview_text = selected_entity_id_ >= 0 ? 
        "Entity #" + std::to_string(selected_entity_id_) : 
        "None";
    
    ImGui::SetNextItemWidth(-1); // Full width
    if (ImGui::BeginCombo("##entity_select", preview_text.c_str())) {
        // Add "None" option
        bool is_selected = (selected_entity_id_ == -1);
        if (ImGui::Selectable("None", is_selected)) {
            selected_entity_id_ = -1;
        }
        if (is_selected) {
            ImGui::SetItemDefaultFocus();
        }
        
        // Add all entities to the list
        for (size_t i = 0; i < entity_count; i++) {
            std::string label = "Entity #" + std::to_string(i);
            bool is_entity_selected = (selected_entity_id_ == (int)i);
            
            if (ImGui::Selectable(label.c_str(), is_entity_selected)) {
                selected_entity_id_ = (int)i;
            }
            
            if (is_entity_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        
        ImGui::EndCombo();
    }
    
    ImGui::Spacing();
    
    // Show details if an entity is selected
    if (selected_entity_id_ >= 0 && selected_entity_id_ < (int)entity_count) {
        ImGui::SeparatorText("Entity Details");
        
        const auto& entity = entities[selected_entity_id_];
        
        // Transform information
        ImGui::Text("Transform:");
        glm::mat4 transform = entity->GetTransform();
        glm::vec3 position = glm::vec3(transform[3]);
        ImGui::Text("  Position: (%.2f, %.2f, %.2f)", position.x, position.y, position.z);
        
        // Scale
        glm::vec3 scale = glm::vec3(
            glm::length(glm::vec3(transform[0])),
            glm::length(glm::vec3(transform[1])),
            glm::length(glm::vec3(transform[2]))
        );
        ImGui::Text("  Scale: (%.2f, %.2f, %.2f)", scale.x, scale.y, scale.z);
        
        ImGui::Spacing();
        
        // Material information
        ImGui::SeparatorText("Material");
        Material mat = entity->GetMaterial();
        
        ImGui::Text("Base Color:");
        ImGui::ColorEdit3("##base_color", &mat.base_color[0], ImGuiColorEditFlags_NoInputs);
        ImGui::Text("  RGB: (%.2f, %.2f, %.2f)", mat.base_color.r, mat.base_color.g, mat.base_color.b);
        
        ImGui::Text("Roughness: %.2f", mat.roughness);
        ImGui::Text("Metallic: %.2f", mat.metallic);
        
        ImGui::Spacing();
        
        // Mesh information
        ImGui::SeparatorText("Mesh");
        if (entity->GetIndexBuffer()) {
            size_t index_count = entity->GetIndexBuffer()->Size() / sizeof(uint32_t);
            size_t triangle_count = index_count / 3;
            ImGui::Text("Triangles: %zu", triangle_count);
            ImGui::Text("Indices: %zu", index_count);
        }
        
        if (entity->GetVertexBuffer()) {
            size_t vertex_size = sizeof(float) * 3; // Assuming pos(3)
            size_t vertex_count = entity->GetVertexBuffer()->Size() / vertex_size;
            ImGui::Text("Vertices: %zu", vertex_count);
        }
        
        ImGui::Spacing();
        
        // BLAS information
        ImGui::SeparatorText("Acceleration Structure");
        if (entity->GetBLAS()) {
            ImGui::Text("BLAS: Built");
        } else {
            ImGui::Text("BLAS: Not built");
        }
    } else {
        ImGui::TextDisabled("No entity selected");
        ImGui::Spacing();
        ImGui::TextWrapped("Hover over an entity to highlight it, then left-click to select. Or use the dropdown above.");
    }
    
    ImGui::End();
}

void Application::OnRender() {
    // Don't render if window is closing
    if (!alive_) {
        return;
    }

    std::unique_ptr<grassland::graphics::CommandContext> command_context;
    core_->CreateCommandContext(&command_context);
    command_context->CmdClearImage(color_image_.get(), { {0.6, 0.7, 0.8, 1.0} });
    command_context->CmdBindRayTracingProgram(program_.get());
    command_context->CmdBindResources(0, scene_->GetTLAS(), grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(1, { color_image_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(2, { camera_object_buffer_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(3, { scene_->GetMaterialsBuffer() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdBindResources(4, { hover_info_buffer_.get() }, grassland::graphics::BIND_POINT_RAYTRACING);
    command_context->CmdDispatchRays(window_->GetWidth(), window_->GetHeight(), 1);
    
    // Render ImGui overlay
    window_->BeginImGuiFrame();
    RenderInfoOverlay();
    RenderEntityPanel();
    window_->EndImGuiFrame();
    
    command_context->CmdPresent(window_.get(), color_image_.get());
    core_->SubmitCommandContext(command_context.get());
}
