#!/bin/bash
# Run QEM + spectral simplification and compare functional maps.
#
# Usage:
#   ./scripts/run_comparison.sh data/spot.obj 500
#   ./scripts/run_comparison.sh data/cutewhale.obj 256 --k 50

set -e

INPUT="$1"
TARGET="$2"
K="${3:---k}"
K_VAL="${4:-100}"

if [ -z "$INPUT" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <input.obj> <target_verts> [--k K]"
    echo "  e.g. $0 data/spot.obj 500"
    echo "       $0 data/cutewhale.obj 256 --k 50"
    exit 1
fi

# Handle --k flag
if [ "$K" = "--k" ] && [ -n "$K_VAL" ]; then
    K_NUM="$K_VAL"
else
    K_NUM=100
fi

# Derive names from input
BASENAME=$(basename "$INPUT" .obj)
SPEC_OUT="out/${BASENAME}_spec_simp_${TARGET}.obj"
QEM_OUT="out/${BASENAME}_qem_simp_${TARGET}.obj"
P_SPEC="out/P_spectral_${BASENAME}_${TARGET}.mtx"
P_QEM="out/P_qem_${BASENAME}_${TARGET}.mtx"
FMAP_PNG="out/fmap_${BASENAME}_${TARGET}.png"

mkdir -p out

echo "=== Spectral Simplification ==="
./cpp/build/spectral_simplify "$INPUT" "$SPEC_OUT" --target "$TARGET" --k "$K_NUM" --verbose --restriction "$P_SPEC"
echo

echo "=== QEM Simplification ==="
python scripts/qem_simplify.py "$INPUT" --output "$QEM_OUT" --target "$TARGET" --restriction "$P_QEM" --no-line-quadric
echo

echo "=== Functional Map Comparison ==="
python scripts/compare_fmaps.py "$INPUT" "$QEM_OUT" "$SPEC_OUT" --P-qem "$P_QEM" --P-spectral "$P_SPEC" -k "$K_NUM" --save-png "$FMAP_PNG"
