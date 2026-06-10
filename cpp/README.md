# C++ Spectral Mesh Simplification

Standalone CLI implementing Lescoat et al. 2020 "Spectral Mesh Simplification."

## Build

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Dependencies (Eigen, Spectra, libigl) are fetched automatically via CMake FetchContent.

## Usage

```bash
./spectral_simplify input.obj output.obj --target 500 --k 30 --verbose
```

Options:
- `--target N` — target vertex count (required)
- `--k K` — number of eigenvectors to preserve (default 30)
- `--verbose` — print progress

## Example

```bash
./spectral_simplify ../../data/spot.obj spot_500.obj --target 500 --k 30 --verbose
```
