"""Builds colab/BBH_Train.ipynb (plain JSON, no nbformat needed). Run: python3 colab/make_notebook.py"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# bbh: train the motor policy

Train on real human Minecraft gameplay (OpenAI VPT contractor data), export one ONNX file, drop it into the mod.

**Before anything else:** `Runtime > Change runtime type > T4 GPU`.

**Run the cells in order. `SMOKE_TEST = True` (default) runs the whole pipeline on 6 segments for about 300 steps
so a broken step shows up in minutes instead of after hours.** Set it to `False` for the real run.

What you get at the end: `bbh_agent_model.zip` containing `policy.onnx`, `spec.json`, `vocab.json`, `goal_blocks.json`,
`model_card.json`. Unzip into `.minecraft/config/bbh/` (or into `mod/model/` before building the mod).
""")

code("""
# 1. Check the GPU and install the two packages Colab does not ship.
import sys, subprocess, torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (training would be very slow)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)
import onnxruntime; print("onnxruntime", onnxruntime.__version__)
""")

code("""
# 2. Get the code. Tries your GitHub repo first; if that fails, it asks you to upload bbh.zip.
import os, sys, zipfile
REPO = "https://github.com/5ivesaw/bbh"
if not os.path.exists("/content/bbh/py/bbh"):
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    if os.system(f"git clone -q {REPO} /content/bbh") != 0 or not os.path.exists("/content/bbh/py/bbh"):
        from google.colab import files
        print("Clone failed (repo not pushed yet?). Upload bbh.zip:")
        up = files.upload()
        zipfile.ZipFile(next(iter(up))).extractall("/content")
sys.path.insert(0, "/content/bbh/py")
from bbh import data, train, export
from bbh.encode import load_spec
spec = load_spec("/content/bbh/contract/spec.json")
print("code ready; image", spec["img_size"], "x", spec["img_size"], "| stride", spec["stride"], "| vocab", spec["vocab_size"])
""")

code("""
# 3. Settings.
from pathlib import Path
SMOKE_TEST   = True        # True: ~10 minute end-to-end check. False: real run.
USE_DRIVE    = True        # checkpoints + exported model survive a Colab disconnect only if they live on Drive
SHARDS_ON_DRIVE = False    # shards are several GB; keep them local (they can be rebuilt) unless Drive has room

N_SEGMENTS   = 6 if SMOKE_TEST else 150     # 5-minute segments (150 = 12.5 hours of play)
STEPS        = 300 if SMOKE_TEST else 20000
BATCH, SEQ   = 32, 32
INDEXES      = ["all_10xx_Jun_29.json", "all_7xx_Apr_6.json", "all_8xx_Jun_29.json"]

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = Path("/content/drive/MyDrive/bbh_work") / ("smoke" if SMOKE_TEST else "full")
else:
    WORK = Path("/content/bbh_work") / ("smoke" if SMOKE_TEST else "full")
LOCAL = Path("/content/bbh_local") / ("smoke" if SMOKE_TEST else "full")
RAW = LOCAL / "raw"
SHARDS = (WORK if SHARDS_ON_DRIVE else LOCAL) / "shards"
for p in (WORK, RAW, SHARDS): p.mkdir(parents=True, exist_ok=True)
print("work:", WORK, "| shards:", SHARDS)
""")

code("""
# 4. Pick segments and download them (jsonl + mp4, 8 in parallel). Missing files are skipped, not fatal.
segments = data.pick_segments(INDEXES, N_SEGMENTS, LOCAL / "index", seed=0)
print(len(segments), "segments chosen, e.g.", segments[0][1])
segments = data.download_many(segments, RAW, workers=8)
print(len(segments), "segments downloaded")
""")

code("""
# 5. Build the item vocabulary from the data itself (top items players actually collect or craft),
#    then convert every segment: decode video -> 64x64, align actions, relabel goals, write shards. Resumable.
import json
jsonls = [RAW / f"{data.seg_name(r)}.jsonl" for _, r in segments]
vocab = data.build_vocab(jsonls, spec["vocab_size"])
(WORK / "vocab.json").write_text(json.dumps(vocab, indent=2))
print(len(vocab), "vocab items:", vocab)
stats = data.convert_many(segments, spec, vocab, RAW, SHARDS, delete_raw=True)
""")

code("""
# 6. Sanity checks BEFORE training. Look at these numbers; if something is off, fix it now, not after training.
import numpy as np, matplotlib.pyplot as plt
done = [json.loads(l) for l in (SHARDS / "done.jsonl").read_text().splitlines() if l.strip()]
print(len(done), "segments converted;", sum(d["steps"] for d in done), "steps =", round(sum(d["steps"] for d in done) / spec["run_hz"] / 3600, 2), "hours")
for f in ["has_pose", "has_inventory", "has_hotbar"]:
    print(f"{f}: {sum(d[f] for d in done)}/{len(done)} segments")
print("mean fraction of steps with a goal:", round(np.mean([d["goal_frac"] for d in done]), 3),
      "| inside GUI (masked):", round(np.mean([d["gui_frac"] for d in done]), 3),
      "| valid camera:", round(np.mean([d["cam_valid_frac"] for d in done]), 3))
print("splits:", {s: sum(d["split"] == s for d in done) for s in ("train", "val")})

d0 = done[0]["name"]
fr = np.load(SHARDS / f"{d0}_frames.npy", mmap_mode="r"); lb = np.load(SHARDS / f"{d0}_labels.npz")
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for a, i in zip(ax.flat, np.linspace(0, len(fr) - 1, 8).astype(int)):
    a.imshow(fr[i]); a.set_title(f"step {i}"); a.axis("off")
plt.show()
print("key frequency in this segment:", dict(zip(spec["keys"], lb["keys"].mean(0).round(3))))
g = np.bincount(lb["goal"], minlength=len(vocab) + 1)
print("goal histogram (id: count):", {(["none"] + vocab)[i]: int(c) for i, c in enumerate(g) if c})
print("YOU SHOULD SEE: recognisable Minecraft frames above, W held a lot, camera bin 10 (centre) most common.")
""")

code("""
# 7. Train. Resumes automatically from the checkpoint in WORK if you re-run after a disconnect.
history = train.run_training(SHARDS, WORK, spec, steps=STEPS, batch_size=BATCH, seq_len=SEQ,
                             eval_every=100 if SMOKE_TEST else 1000, save_every=100 if SMOKE_TEST else 500,
                             max_minutes=None if SMOKE_TEST else 600)
""")

code("""
# 8. Training curves.
import matplotlib.pyplot as plt
tr = [(h["step"], h["train"]["loss"]) for h in history if "train" in h]
va = [(h["step"], h["val"]["loss"]) for h in history if "val" in h]
plt.plot(*zip(*tr), label="train")
if va: plt.plot(*zip(*va), label="validation", marker="o")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show()
last = next((h["val"] for h in reversed(history) if "val" in h), None)
print(last)
""")

code("""
# 9. Export to ONNX, check it against PyTorch, time it, zip it, download it.
zip_path, meta = export.export_onnx(WORK / "ckpt.pt", WORK / "bbh_agent_model", spec, vocab,
                                    goal_blocks_path="/content/bbh/contract/goal_blocks.json")
print(meta)
from google.colab import files
files.download(str(zip_path))
""")

md("""
## After the download

1. Unzip into `<minecraft folder>/config/bbh/` (create the folder), or into `mod/model/` before building so it ships inside the jar.
2. Build the mod (`mod/README` in the repo) and start Minecraft with Fabric API (and Baritone if you want delegation).
3. In a world: `/bbh dumpframe` then open `config/bbh/frame.png` to confirm the network sees correct colors, then `/bbh goal log` and `/bbh start`.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "BBH_Train.ipynb", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "BBH_Train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
