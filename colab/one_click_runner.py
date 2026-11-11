# BBH — ONE CELL: click Run once. Defaults to a quick end-to-end smoke test.
# After success, change SMOKE_TEST to False and run this SAME cell again.
SMOKE_TEST = True
USE_DRIVE = True  # checkpoints survive disconnects; Google may request authorization
CACHE_SHARDS_ON_DRIVE = True  # trades Drive space (~GBs) for safe reconnect/resume
N_SEGMENTS = 6 if SMOKE_TEST else 150
TRAIN_STEPS = 30 if SMOKE_TEST else 20000
BATCH_SIZE = 8 if SMOKE_TEST else 32
SEQ_LEN = 16 if SMOKE_TEST else 32
INDEXES = ["all_10xx_Jun_29.json", "all_7xx_Apr_6.json", "all_8xx_Jun_29.json"]

# The notebook generator replaces __ARCHIVE_PLACEHOLDER__ with embedded project
# code. No repo clone or separate upload is required in the generated notebook.
import base64, importlib.util, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from time import monotonic

started = monotonic()
source_archive = __ARCHIVE_PLACEHOLDER__
root = Path("/content/bbh")
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(source_archive))) as z:
    z.extractall(root)
for name in list(sys.modules):
    if name == "bbh" or name.startswith("bbh."):
        del sys.modules[name]  # reload bundled source when running again
sys.path.insert(0, str(root / "py"))

missing = [package for module, package in (("cv2", "opencv-python-headless"),
           ("onnx", "onnx"), ("onnxruntime", "onnxruntime"))
           if importlib.util.find_spec(module) is None]
if missing:
    print("Installing", missing, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
import numpy as np
import torch
import matplotlib.pyplot as plt
from bbh import data, train, export
from bbh.encode import load_spec

if not torch.cuda.is_available():
    raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > L4/T4 GPU, then rerun this cell.")
print("GPU", torch.cuda.get_device_name(0), "| PyTorch", torch.__version__, flush=True)
spec = load_spec(root / "contract" / "spec.json")
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    work = Path("/content/drive/MyDrive/bbh_work") / ("smoke" if SMOKE_TEST else "full")
else:
    work = Path("/content/bbh_work") / ("smoke" if SMOKE_TEST else "full")
local = Path("/content/bbh_local") / ("smoke" if SMOKE_TEST else "full")
raw, shards = local / "raw", local / "shards"
for path in (work, raw, shards):
    path.mkdir(parents=True, exist_ok=True)
done_file, vocab_file = shards / "done.jsonl", work / "vocab.json"
cache = work / "shard_cache" if USE_DRIVE and CACHE_SHARDS_ON_DRIVE else None
if cache:
    cache.mkdir(parents=True, exist_ok=True)
    # Restore to fast local storage for training (random reads from mounted Drive are slow).
    if not done_file.exists() and (cache / "done.jsonl").exists():
        shutil.copy2(cache / "done.jsonl", done_file)
    if done_file.exists():
        for entry in (json.loads(l) for l in done_file.read_text().splitlines() if l.strip()):
            for suffix in ("_frames.npy", "_labels.npz"):
                filename = entry["name"] + suffix
                if not (shards / filename).exists() and (cache / filename).exists():
                    shutil.copy2(cache / filename, shards / filename)

def read_done():
    records = [json.loads(line) for line in done_file.read_text().splitlines() if line.strip()] if done_file.exists() else []
    unique = {rec["name"]: rec for rec in records}
    for name in unique:
        for suffix in ("_frames.npy", "_labels.npz"):
            if not (shards / (name + suffix)).exists():
                raise RuntimeError(f"Incomplete shard: {name + suffix}. Repair the saved shards before resuming.")
    return unique

def persist_shards():
    if cache is None or not done_file.exists():
        return
    def atomic_copy(src, dest):
        temporary = dest.with_name(dest.name + ".part")
        shutil.copy2(src, temporary)
        os.replace(temporary, dest)
    # Save files first, then the manifest; a disconnect cannot leave a manifest
    # claiming a shard exists when its actual files have not been copied yet.
    for name in read_done():
        for suffix in ("_frames.npy", "_labels.npz"):
            filename = name + suffix
            if (not (cache / filename).exists() or
                    (cache / filename).stat().st_size != (shards / filename).stat().st_size):
                atomic_copy(shards / filename, cache / filename)
    atomic_copy(done_file, cache / "done.jsonl")

done = read_done()
if done and not vocab_file.exists():
    raise RuntimeError("Saved shards have no vocab.json. Restore the old vocabulary before resuming.")
print("Mode:", "SMOKE TEST" if SMOKE_TEST else "FULL", "| converted:", len(done), "/", N_SEGMENTS, flush=True)

if len(done) < N_SEGMENTS:
    candidates = data.pick_segments(INDEXES, None, local / "index", seed=0)
    print("Index candidates:", len(candidates), "| URL example:", data.segment_urls(*candidates[0])[0], flush=True)
    vocab = json.loads(vocab_file.read_text()) if vocab_file.exists() else []
    if done and not vocab:
        raise RuntimeError("Cannot resume shards with an empty vocabulary.")
    attempted = set(done)
    if not vocab:
        # Missing recordings in the public index are replaced automatically.
        seed_n = min(6 if SMOKE_TEST else 12, N_SEGMENTS)
        initial = data.download_first_valid(candidates, raw, seed_n, workers=4,
                 skip_names=attempted, max_attempts=max(60, seed_n * 4))
        attempted.update(data.seg_name(rel) for _, rel in initial)
        vocab = data.build_vocab([raw / (data.seg_name(rel) + ".jsonl") for _, rel in initial],
                                 spec["vocab_size"])
        if not vocab:
            raise RuntimeError("Downloaded VPT action files contained no inventory gains. Stopped before training.")
        vocab_file.write_text(json.dumps(vocab, indent=2))
        print("Vocabulary:", len(vocab), "items | sample:", vocab[:12], flush=True)
        data.convert_many(initial, spec, vocab, raw, shards, delete_raw=True)
        done = read_done()
        persist_shards()
        print("Converted initial recordings:", len(done), "/", N_SEGMENTS, flush=True)

    # Download/convert/delete one recording at a time to avoid filling Colab disk.
    extra_tries, max_tries = 0, max(60, N_SEGMENTS * 4)
    for base, rel in candidates:
        if len(done) >= N_SEGMENTS or extra_tries >= max_tries:
            break
        name = data.seg_name(rel)
        if name in attempted:
            continue
        attempted.add(name)
        extra_tries += 1
        converted = data.convert_many([(base, rel)], spec, vocab, raw, shards, delete_raw=True)
        if converted:
            done = read_done()
            persist_shards()
            print(f">>> converted {len(done)}/{N_SEGMENTS} ({(monotonic() - started) / 60:.1f} min)", flush=True)
    if len(done) < N_SEGMENTS:
        raise RuntimeError(f"Only {len(done)}/{N_SEGMENTS} segments converted after {extra_tries} extra attempts. "
                           "Training did NOT start. Check the last download/conversion error above.")
else:
    vocab = json.loads(vocab_file.read_text())
    if not vocab:
        raise RuntimeError("Saved vocabulary is empty; restore the real vocabulary before resuming.")

# Sanity-check converted data BEFORE committing GPU time to training.
records = list(read_done().values())
print("DATA READY:", len(records), "segments;", sum(r["steps"] for r in records), "steps =",
      round(sum(r["steps"] for r in records) / spec["run_hz"] / 3600, 2), "hours", flush=True)
print("Pose/inventory/hotbar coverage:",
      *[f"{k}: {sum(bool(r[k]) for r in records)}/{len(records)}" for k in
        ("has_pose", "has_inventory", "has_hotbar")], flush=True)
print("Splits:", {s: sum(r["split"] == s for r in records) for s in ("train", "val")}, flush=True)
name = records[0]["name"]
frames = np.load(shards / (name + "_frames.npy"), mmap_mode="r")
labels = np.load(shards / (name + "_labels.npz"))
if len(frames) < 64 or np.asarray(frames[:8], dtype=np.float32).std() < 2:
    raise RuntimeError("Frames are too short or near-blank. Stopped before training; inspect the sample.")
print("Key frequency:", dict(zip(spec["keys"], labels["keys"].mean(0).round(3))), flush=True)
fig, axes = plt.subplots(1, 4, figsize=(12, 2.5))
for ax, i in zip(axes, np.linspace(0, len(frames) - 1, 4).astype(int)):
    ax.imshow(frames[i]); ax.set_title(f"frame {i}"); ax.axis("off")
plt.tight_layout(); plt.show(); plt.close(fig)

print("TRAINING, with checkpoints saved to", work, flush=True)
history = train.run_training(shards, work, spec, steps=TRAIN_STEPS,
           batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
           eval_every=100000 if SMOKE_TEST else 1000,
           save_every=30 if SMOKE_TEST else 500,
           max_minutes=None if SMOKE_TEST else 600)
ck = torch.load(work / "ckpt.pt", map_location="cpu", weights_only=False)
if ck["step"] < TRAIN_STEPS:
    raise RuntimeError(f"Checkpoint saved at {ck['step']}/{TRAIN_STEPS} steps. "
                       "Colab time budget reached. Rerun THIS SAME cell to resume, then export.")
tr = [(h["step"], h["train"]["loss"]) for h in history if "train" in h]
va = [(h["step"], h["val"]["loss"]) for h in history if "val" in h]
if tr:
    plt.plot(*zip(*tr), label="train")
    if va: plt.plot(*zip(*va), label="validation")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show(); plt.close()

zip_path, meta = export.export_onnx(work / "ckpt.pt", work / "bbh_agent_model",
                                    spec, vocab, goal_blocks_path=root / "contract" / "goal_blocks.json")
print("SUCCESS — model ZIP:", zip_path, "| ONNX/PyTorch error:", meta["onnx_max_abs_diff"],
      "| elapsed minutes:", round((monotonic() - started) / 60, 1), flush=True)
from google.colab import files
files.download(str(zip_path))
