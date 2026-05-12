# `kor_deepcv_large` — KOR + 21M-parameter Cα CV

The 80 MB `deepcv_allca.pt` model is **not shipped** with the repo — it's
too large for GitHub. Place the file at this path before running:

```
benchmarks/kor_deepcv_large/deepcv_allca.pt
```

Recovery options:

1. **Re-train.** Use `benchmarks/kor_a100/deep_cv/02_train_allca.py` on a
   GPU with enough VRAM. Takes ~3 s on an RTX 5090 to converge the Fisher
   loss after `01_gather_allca.py` produced training data.
2. **Download from your own storage** (S3, Zenodo, etc.) — keep a copy
   anywhere persistent and `cp` / `wget` it into place.

Then run as part of the suite:

```bash
python benchmarks/run_suite.py --only kor_deepcv_large
```

Reference measurement (RTX 5090, smoke timing): **~6 steps/s, ~1 ns/day**
on a 71 873-atom system. The 150× slowdown vs `kor_deepcv_small` is the
real cost of running a 21M-parameter torch model + autograd backward on
every MD step. Useful for sizing the upper bound of PyTorch-CV overhead.
