cmake_minimum_required(VERSION 3.18)
project(spectral_simplify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

# Spectra (header-only eigenvalue solver)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG v1.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spectra)

# libigl (header-only geometry processing)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libigl)

add_executable(spectral_simplify
    src/main.cpp
    src/spectral_simplify.cpp
    src/spectral_cost.cpp
    src/mesh_adjacency.cpp
)

target_include_directories(spectral_simplify PRIVATE src)
target_link_libraries(spectral_simplify PRIVATE
    Eigen3::Eigen
    igl::core
)
target_include_directories(spectral_simplify PRIVATE ${spectra_SOURCE_DIR}/include)
