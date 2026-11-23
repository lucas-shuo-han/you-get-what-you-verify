"""Push one final-experiment run (§4m) as a private Kaggle kernel: python tools/kaggle_final_push.py G SEED MULT [--user U]
Builds notebooks/final/G_sSEED_xMULT.ipynb via build_colab_mix_notebook.py --kaggle, k=0 eval, connectivity on Kaggle."""
import json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import kaggle_push as K  # noqa: E402
g, seed, mult = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = f"{g}_s{seed}_x{mult}"; nb = K.ROOT / f"notebooks/final/{tag}.ipynb"; nb.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(K.ROOT / "tools/build_colab_mix_notebook.py"), "--kaggle", "--group", g, "--seed", str(seed),
                "--mult", str(mult), "--max_budget", "0", "--max_steps", "292", "--out", str(nb)], check=True)
user = K.username(); slug = f"brickexp-final-{g.lower()}{seed}x{mult}"
k = K.STAGE / f"kernel_final_{tag}"; shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
shutil.copy(nb, k / "run.ipynb")
(k / "kernel-metadata.json").write_text(json.dumps({"id": f"{user}/{slug}", "title": slug, "code_file": "run.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [f"{user}/{K.DS_SLUG}"], "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
rc, out = K.sh("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(11 * 3600), check=False)
print("PUSH", slug, "ok" if rc == 0 and "error" not in out.lower() else "FAILED: " + out.strip()[-200:])
