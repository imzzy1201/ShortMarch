# ShortMarch

## Description

This is the repository for course project the Advanced Computer Graphics instructed by *Li Yi* at IIIS, Tsinghua University. 

The code is based on ShortMarch written by *He Li* (TA for 2025 Fall Semester), which contains a simple framework for GPU rendering downgraded from [LongMarch](https://github.com/LazyJazzDev/LongMarch/tree/main) by *Zijian Lyu*.

This project presents the design and implementation of a physically based hardware rendering system capable of producing realistic images of complex 3D scenes. The system aims to faithfully model light transport and material interactions while maintaining practical performance through carefully designed rendering techniques and optimizations.

Our rendering system implements a range of physically based rendering features. In particular, we implement:

- A basic path tracing pipeline that simulates physically accurate light transport.
- Principled BSDF material models to achieve realistic and consistent surface appearance.
- Support for multiple types of light sources, including point lights, area lights, directional (sun) lights, and environment lighting.
- Volumetric rendering for participating media, enabling effects such as fog, smoke, and particle-based phenomena.
- Camera and temporal effects, including depth of field and motion blur, to enhance visual realism.

Using this rendering system, we construct and render a variety of scenes that showcase both the visual quality and flexibility of our implementation.

## How to build

_The following section is from Shortmarch's README.md_ 

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

_The following section is the extra requirement to build our repo._

### Download the objects and materials files

Our repository uses objects provided in LongMarch's assets, but also uses some additional custom objects.

You can download the objects from https://cloud.tsinghua.edu.cn/d/968e11c547684ae8b5a4/ ,and use them to replace the `external\LongMarch\assets\meshes` folder.

After that, you can generate different scenes by modifying the code in `app.cpp`.
## Implemented Features

### Path Tracing Core

- Progressive path tracing supporting diffuse and specular,transmissive transport

- Create a custom scene with tidiness and attractiveness


### Materials & BSDFs

- Transmissive materials with refraction

- Principled BSDF based on microfacet theory

- Multi-layer materials, including clearcoat-style layering

### Texture

- Color texture mapping with material images

### Importance Sampling
- Importance sampling with Russian Roulette,mutiple importance sampling for BSDFs
### Sampling & Anti-Aliasing
Combined sampling strategies for variance reduction
### Volumetric & Subsurface Effects

- Homogeneous volume rendering with free-flight sampling

- Subsurface scattering (channel-independent approximation)

- Volumetric alpha shadows

### Special Visual Effects

- Motion blur, depth of field(by setting properties of entities and camera)

- Alpha shadow
### Lighting

- Point lights and area lights with shadow testing

- Environment lighting using HDR skyboxes

- Additionally,sunlight with given angle

### Anti-aliasing
- Anti-aliasing via stochastic pixel jitter(already implemented in ShortMarch)

_The following parts is from ShortMarch's README.md_.

## Project Structure

```
src/
├── main.cpp              # Application entry point
├── app.h/app.cpp         # Main application class with rendering loop
├── Scene.h/Scene.cpp     # Scene manager (TLAS, materials buffer)
├── Entity.h/Entity.cpp   # Entity class (mesh, BLAS, transform)
├── Film.h/Film.cpp       # Film class for progressive accumulation
├── Material.h            # Material structure for PBR properties
└── shaders/
    └── shader.hlsl       # Ray tracing shaders (raygen, miss, closest hit)
```
## Key Features
#### 1. Scene-Based Architecture
- **Scene Management**: The `Scene` class manages multiple entities and builds the Top-Level Acceleration Structure (TLAS)
- **Entity System**: Each `Entity` contains a mesh (loaded from `.obj` files), a material, and a transform matrix
- **Materials**: PBR materials with complex member variables and custom material images,satisfying the requirement of principled BSDF.

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
- **Pixel-Perfect Picking**: Uses a GPU-rendered entity ID buffer for accurate entity detection under the cursor
- **Hover Highlighting**: Entities glow yellow when the cursor hovers over them
- **Click Selection**: Left-click on an entity to select it and view details in the right panel

#### 4. Progressive Accumulation (Film Class)
- **Automatic Accumulation**: When camera is stationary (camera mode disabled), samples accumulate over time
- **High-Quality Rendering**: Progressive refinement produces noise-free images with more samples
- **Smart Reset**: Accumulation automatically resets when camera movement stops
- **Real-time Feedback**: Sample count displayed in UI shows accumulation progress

#### 5. Pixel Inspector
- **Real-time Color Sampling**: Shows RGB values of the pixel under the cursor
- **Original Color Display**: Values shown are before highlighting is applied (matches saved screenshots)
- **Multiple Formats**: Both normalized float (0.0-1.0) and 8-bit (0-255) values
- **Color Preview**: Visual color swatch shows the exact pixel color
- **Mouse Position**: Displays current cursor coordinates

#### 6. Screenshot Capture
- **Ctrl+S Shortcut**: Save accumulated output as PNG image
- **Automatic Naming**: Timestamped filenames (e.g., `screenshot_20251101_225009.png`)
- **Full Path Logging**: Console shows complete absolute path where image is saved
- **Pure Rendering**: Saved images exclude UI overlays and hover highlights
- **High Quality**: Captures the fully accumulated, noise-free render

#### 7. ImGui Interface
Two non-collapsible panels appear in inspection mode:
- **Left Panel** (Scene Information):
  - Camera position, direction, yaw, pitch
  - Speed and sensitivity settings
  - Entity count, material count, total triangles
  - Hovered and selected entity IDs
  - **Pixel Inspector**: Mouse position and RGB color values
  - Render information (resolution, backend, device)
  - Accumulation status and sample count
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

4. **Inspect Pixels**:
   - Hover over any part of the rendered image
   - View RGB color values in the Pixel Inspector section
   - Values shown are the original rendered colors (before highlighting)

5. **Hide UI** (inspection mode only):
   - Hold **Tab** key to temporarily hide all UI panels
   - Useful for taking clean screenshots or viewing full render

6. **Save Screenshots**:
   - Press **Ctrl+S** to save the current accumulated output as PNG
   - Images saved with timestamp in filename
   - Console shows full path where image is saved
   - Saved images are clean (no UI, no highlights)


