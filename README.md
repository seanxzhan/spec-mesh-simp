# Spectral Mesh Simplification

## Comparison

QEM simplification:
```bash
python scripts/qem_simplify.py data/cutewhale.obj --output cutewhale_qem_simp_256.obj --target 256 --restriction out/P_qem_cutewhale.mtx --no-line-quadric
```

Spectral simplification:
```bash
./cpp/build/spectral_simplify data/cutewhale.obj cutewhale_spec_simp_256.obj --target 256 --k 100 --verbose --restriction out/P_spectral_cutewhale.mtx
```

Compare functional maps:
```bash
python scripts/compare_fmaps.py data/cutewhale.obj cutewhale_qem_simp_256.obj cutewhale_spec_simp_256.obj --P-qem out/P_qem_cutewhale.mtx --P-spectral out/P_spectral_cutewhale.mtx -k 100 --save-png out/fmap_cutewhale.png
```