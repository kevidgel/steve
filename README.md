# steve - a pathtracer

**15-468 Final Project**

![renderer](steve.png)

### Technical Award Winner for 15-468 Spring 2025 

[Link [https://graphics.cs.cmu.edu/courses/15-468/rendering_competition.html]](https://graphics.cs.cmu.edu/courses/15-468/rendering_competition.html)

## About

Steve is a GPU-accelerated photorealistic rendering engine implementing a physically-based pathtracer with advanced sampling techniques. It leverages NVIDIA CUDA and OptiX for real-time ray tracing on modern GPUs, combined with ReSTIR for variance reduction and improved image quality.

## Building

### Prerequisites

- NVIDIA GPU with CUDA compute capability (e.g., RTX series)
- CUDA 12.0+ compatible driver
- OptiX 7.0+ SDK
- CMake 3.25 or higher
- C++ compiler with C++20 support (clang or g++)
- vcpkg installed and `VCPKG_ROOT` environment variable set

### Build Steps

1. **Set up vcpkg root** (if not already set):
   ```bash
   export VCPKG_ROOT=/path/to/vcpkg
   ```

2. **Generate build files**:
   ```bash
   cmake --preset default # Check CMakeUserPresets.json to set appropraite build variables
   ```

3. **Build the project**:
   ```bash
   cmake --build build
   ```

The executable will be available at `build/src/app/steve`.

## Usage

Run the pathtracer with a scene configuration file:

```bash
./steve path/to/scene.json
```
A sample scene configuration can be found in `/scene.json`, which renders `fireplace_room`. The model can be downloaded via `download_scenes.sh`.

> [!WARNING]
> This project uses OpenGL, which may automatically pick the integrated graphics card instead of your NVIDIA discrete graphics card. On Linux, a workaround is by prepending these environment variables: `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` when executing the binary.

The scene JSON file specifies:
- Camera position, field of view, and viewport settings
- 3D models to load (OBJ or glTF format)
- Materials and their parameters
- Light sources
- Rendering parameters (samples per pixel, max bounces, etc.)

### Interactive Controls

Once running, use the ImGui interface to:
- Adjust camera position and parameters in real-time
- Modify material properties
- Control rendering settings
- Export rendered frames


Alternatively, keyboard controls can be used. WASD control the position of the camera, and arrow keys control the direction of the viewport.

## Known Bugs
- For scenes with 0 emissive primitives, integrators other than the base Direct and BRDF integrators will crash the program. This is because it creates a lights mesh of 0 objects.
- I recently updated the OWL dependency, and it has slightly different resource handling than the previous version I used, which may lead to some assertions failing during exit. 
