#include "spectral_simplify.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " input.obj output.obj --target N [options]\n"
              << "Options:\n"
              << "  --target N      Target vertex count (required)\n"
              << "  --k K           Number of eigenvectors to preserve (default: 30)\n"
              << "  --restriction F Save restriction matrix P in Matrix Market format\n"
              << "  --verbose       Print progress\n";
}

void save_matrix_market(const std::string& path, const Eigen::SparseMatrix<double>& M) {
    std::ofstream f(path);
    f << "%%MatrixMarket matrix coordinate real general\n";
    f << M.rows() << " " << M.cols() << " " << M.nonZeros() << "\n";
    for (int k = 0; k < M.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            f << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) { print_usage(argv[0]); return 1; }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int target = -1;
    int K = 30;
    bool verbose = false;
    std::string restriction_path;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--target" && i + 1 < argc) { target = std::stoi(argv[++i]); }
        else if (arg == "--k" && i + 1 < argc) { K = std::stoi(argv[++i]); }
        else if (arg == "--restriction" && i + 1 < argc) { restriction_path = argv[++i]; }
        else if (arg == "--verbose") { verbose = true; }
        else { std::cerr << "Unknown option: " << arg << std::endl; return 1; }
    }

    if (target <= 0) { std::cerr << "Error: --target required\n"; return 1; }

    // Load mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(input_path, V, F)) {
        std::cerr << "Failed to read: " << input_path << std::endl;
        return 1;
    }
    std::cout << "Input: " << V.rows() << " verts, " << F.rows() << " faces" << std::endl;

    if (target >= (int)V.rows()) {
        std::cerr << "Target >= input vertices, nothing to do." << std::endl;
        return 1;
    }

    // Simplify
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = spectral_simplify(V, F, target, K, verbose);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Output: " << result.V.rows() << " verts, " << result.F.rows() << " faces"
              << " (" << dt << "s)" << std::endl;

    // Save mesh
    if (!igl::writeOBJ(output_path, result.V, result.F)) {
        std::cerr << "Failed to write: " << output_path << std::endl;
        return 1;
    }
    std::cout << "Saved: " << output_path << std::endl;

    // Save restriction matrix P
    if (!restriction_path.empty()) {
        save_matrix_market(restriction_path, result.P);
        std::cout << "Saved P: " << restriction_path << " (" << result.P.rows()
                  << "x" << result.P.cols() << ", " << result.P.nonZeros() << " nnz)" << std::endl;
    }

    return 0;
}
