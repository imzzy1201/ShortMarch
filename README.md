# ShortMarch

## Description

This is the official repository for the Advanced Computer Graphics instructed by *Li Yi* at IIIS, Tsinghua University. 

This project contains a simple framework for GPU rendering downgraded from [LongMarch](https://github.com/LazyJazzDev/LongMarch/tree/main) by *Zijian Lyu*.

The demo code is written by *He Li* (TA for 2025 Fall Semester), feel free to contact him if you have any questions.

## Honor Code

You are expected to uphold the principles of academic integrity and honesty in all your work related to this repository. Any form of academic dishonesty, including but not limited to plagiarism, cheating, or unauthorized collaboration, is strictly prohibited and may result in severe consequences. **Any direct copy and paste of code (even from the `external/` code this repository referred) or using AI tools to generate code with knowledge of the subject matter will be considered as cheating**. You are free to read any reference materials, including books, articles, and online resources, to enhance your understanding of the subject matter.

By accessing and using this repository, you acknowledge that you have read, understood, and agreed to abide by this Honor Code. If you do not agree to these terms, you must refrain from using this repository.

## How to build

We recommend using [Visual Studio](https://visualstudio.microsoft.com/) as the IDE for building this project.

### Step 0: Prerequisites

- [vcpkg](https://github.com/microsoft/vcpkg): The C++ package manager. Clone the vcpkg repo to anywhere you like, we will refer tha vcpkg path as
  `<VCPKG_ROOT>` in the following instructions (the path ends in `vcpkg`, not its parent directory).
- [MSVC with Windows SDK (version 10+)](https://visualstudio.microsoft.com/downloads/): We usually install this via Visual Studio installer. You should select the following workloads during installation:
  - Desktop development with C++

  Then everything should be installed automatically.
- [[optional] Python3](https://python.org): We provide python package with pybind11. Such functionality requires Python3 installation. You may install anywhere you like (System-wide, User-only, Conda, Homebrew, etc.). We will refer the python executable path as `<PYTHON_EXECUTABLE_PATH>` in the following instructions.
- [[optional] Vulkan SDK](https://vulkan.lunarg.com/sdk/home): Vulkan is the latest cross-platform graphics API. Since D3D12 is available on Windows, this is optional. Install the SDK [Caution: not the Runtime (RT)] via the official **SDK installer**. You should be able to run `vulkaninfo` command in a new terminal after installation. **No optional components are needed for this project**.
- [[optional] CUDA Toolkit](https://developer.nvidia.com/cuda-downloads): CUDA is optional, however, some functions such as most of the GPU-accelerated physics simulation features will require CUDA. Install the toolkit with the official **exe (local)** installer. You should be able to run `nvcc --version` command in a new terminal after installation.

- ### Step 1: Clone the repo

- Clone this repo with submodules:
  ```bash
  git clone --recurse-submodules
  ```
  or
- Clone without submodules:
  ```bash
  git clone <this-repo-url>
  ```
  Then initialize and update the submodules (in the root directory of this repo):
  ```bash
  git submodule update --init --recursive
  ```

### Step 2: CMake Configuration

In Visual Studio, open the `Project` -> `CMake Settings for Project` menu, and modify the `CMake toolchain file` to: `<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake`.

In this process, the CMake script will check whether you have installed Vulkan SDK and CUDA Toolkit, and configure the build options accordingly.

### Step 3: Build and Run

Now you can build and run the project in Visual Studio as usual, selecting the desired target (`ShortMarchDemo.exe` for the demo we provided).

## Bug Shooting

### CMake Configuration Issues

Make sure that you have set the `CMake toolchain file` correctly to `<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake`. After any change to the configuration, remember to clean the CMake cache (via `Project` -> `CMake Cache` -> `Delete Cache and Reconfigure` menu in Visual Studio) and reconfigure the project.

### Vulkan Validation Layer Error

If you encounter the following error when running the application:
```
validation layer (ERROR): loader_get_json: Failed to open JSON file </path/to/a/json>
```
where `/path/to/a/json` is a non-existent file, it indicates that the Vulkan validation layers are trying to load a configuration file that does not exist on your system. Hopefully, the </path/to/a/json> is related to your Steam or Epic Games installation. To resolve this issue, you can try the following steps:
1. Press `Win + R` and type `regedit` to open the Registry Editor.
2. Try to find the `</path/to/a/json>` under:
	- `HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\Vulkan\ImplicitLayers`
	- `HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\Vulkan\ExplicitLayers`
	- `HKEY_CURRENT_USER\SOFTWARE\Khronos\Vulkan\ImplicitLayers`
	- `HKEY_CURRENT_USER\SOFTWARE\Khronos\Vulkan\ExplicitLayers`.
3. Delete the entry that points to the non-existent JSON file and restart your program.

## Getting Started with the Ray Tracing Demo

The `src/` directory contains a minimalistic interactive ray tracing demo that showcases hardware-accelerated ray tracing using the LongMarch framework. This demo features a scene-based architecture with entity management, interactive camera controls, and an ImGui-based inspection interface.

In your own project, you could either start from this demo or build from scratch. You could modify any file in the `src/` directory to fit your needs.

### Project Structure

```
src/
├── main.cpp              # Application entry point
├── app.h/app.cpp         # Main application class with rendering loop
├── Scene.h/Scene.cpp     # Scene manager (TLAS, materials buffer)
├── Entity.h/Entity.cpp   # Entity class (mesh, BLAS, transform)
├── Material.h            # Material structure for PBR properties
└── shaders/
    └── shader.hlsl       # Ray tracing shaders (raygen, miss, closest hit)
```

### Key Features

#### 1. Scene-Based Architecture
- **Scene Management**: The `Scene` class manages multiple entities and builds the Top-Level Acceleration Structure (TLAS)
- **Entity System**: Each `Entity` contains a mesh (loaded from `.obj` files), a material, and a transform matrix
- **Materials**: Simple PBR materials with base color, roughness, and metallic properties

#### 2. Interactive Camera Controls
The demo supports two modes:
- **Camera Mode** (right-click to enable):
  - `W/A/S/D` - Move forward/left/backward/right
  - `Space/Shift` - Move up/down
  - Mouse - Look around (cursor hidden)
  
- **Inspection Mode** (right-click to disable camera):
  - Mouse - Hover over entities to highlight them
  - Left-click - Select entity for detailed inspection
  - UI panels display camera, scene, and entity information

#### 3. Entity Highlighting and Selection
- **Hover Highlighting**: Entities glow yellow when the cursor hovers over them
- **Click Selection**: Left-click on an entity to select it and view details in the right panel
- **Ray-Sphere Intersection**: CPU-side ray casting matches GPU rendering for accurate picking

#### 4. ImGui Interface
Two non-collapsible panels appear in inspection mode:
- **Left Panel** (Scene Information):
  - Camera position, direction, yaw, pitch
  - Speed and sensitivity settings
  - Entity count, material count, total triangles
  - Hovered and selected entity IDs
  - Render information (resolution, backend, device)
  - Controls hint
  
- **Right Panel** (Entity Inspector):
  - Dropdown to select any entity
  - Transform information (position, scale)
  - Material properties (base color, roughness, metallic)
  - Mesh statistics (triangles, vertices, indices)
  - BLAS build status

### How to Use

1. **Build and Run**:
   ```bash
   # In Visual Studio, select target: ShortMarchDemo.exe
   # Press F5 to build and run
   ```

2. **Navigate the Scene**:
   - Start in inspection mode (cursor visible)
   - Right-click to enable camera mode and fly around
   - Right-click again to return to inspection mode

3. **Inspect Entities**:
   - Move cursor over objects to see them highlight in yellow
   - Left-click to select an entity
   - View detailed information in the right panel
   - Or use the dropdown menu to select entities manually

### Code Architecture

#### Application Class (`app.h/app.cpp`)
The main application class manages:
- Graphics core initialization (D3D12 or Vulkan)
- Window creation and event handling
- Camera state and controls
- Scene rendering and entity interaction
- ImGui interface rendering

Key methods:
- `OnInit()` - Initialize graphics, create scene, load entities
- `OnUpdate()` - Process input, update hover detection, upload GPU buffers
- `OnRender()` - Execute ray tracing, render ImGui overlays
- `OnClose()` - Clean up resources
- `UpdateHoveredEntity()` - CPU-side ray casting for entity picking

#### Scene Class (`Scene.h/Scene.cpp`)
Manages the scene graph:
- `AddEntity()` - Add entities to the scene
- `BuildAccelerationStructures()` - Build TLAS from all entity BLAS
- `UpdateMaterialsBuffer()` - Upload materials to GPU
- `GetTLAS()` - Get the acceleration structure for rendering

#### Entity Class (`Entity.h/Entity.cpp`)
Represents individual objects:
- `LoadMesh()` - Load geometry from `.obj` files
- `BuildBLAS()` - Create Bottom-Level Acceleration Structure
- Material and transform properties

#### Shader (`shaders/shader.hlsl`)
HLSL ray tracing shaders:
- `RayGenMain` - Generate primary rays from camera
- `MissMain` - Sky gradient for missed rays
- `ClosestHitMain` - Shading with material properties and hover highlighting

### Adding New Entities

To add new objects to the scene, edit `Application::OnInit()` in `app.cpp`:

```cpp
// Example: Add a new red sphere
auto red_sphere = std::make_shared<Entity>(
    "meshes/preview_sphere.obj",                    // Mesh path
    Material(glm::vec3(1.0f, 0.0f, 0.0f), 0.3f, 0.0f),  // Red, smooth, non-metallic
    glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 1.0f, 0.0f))  // Position
);
scene_->AddEntity(red_sphere);
```

After adding entities, remember to call `scene_->BuildAccelerationStructures()`.

### Customizing Materials

Materials use a simple PBR model:
```cpp
Material(
    glm::vec3(r, g, b),  // Base color (0.0 to 1.0)
    roughness,            // Surface roughness (0.0 = smooth, 1.0 = rough)
    metallic              // Metallic factor (0.0 = dielectric, 1.0 = metal)
);
```

### Technical Details

- **Acceleration Structures**: Uses hardware ray tracing with BLAS per entity and a single TLAS
- **Resource Bindings**:
  - Space 0: Acceleration Structure (TLAS)
  - Space 1: Output image (UAV)
  - Space 2: Camera info (constant buffer)
  - Space 3: Materials (structured buffer)
  - Space 4: Hover info (constant buffer)
- **Ray-Sphere Intersection**: Approximate bounding sphere per entity for fast CPU picking
- **Coordinate Systems**: Matches GPU ray generation for pixel-perfect hover detection

### Known Limitations

- **Approximate Hover Detection**: Uses bounding spheres instead of precise mesh intersection
- **Simple Lighting**: Placeholder normal (up vector) for diffuse shading
- **Static Scenes**: Animation requires manual `UpdateInstances()` calls
- **Single Window**: ImGui context supports only one window at a time

### Further Development

To extend this demo, consider:
- **Proper Normal Calculation**: Compute normals from triangle geometry in shaders
- **Advanced Materials**: Implement full PBR with textures and normal mapping
- **Dynamic Scenes**: Add animation system with automatic TLAS updates
- **Better Picking**: Use GPU ray tracing for exact entity selection
- **Shadows and Reflections**: Cast secondary rays for more realistic lighting