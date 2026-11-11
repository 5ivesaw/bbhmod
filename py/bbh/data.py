"""Download VPT contractor segments and convert them into training shards.

Real file layout (verified against the live VPT index):
  index:   https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/<name>.json
           -> {"basedir": ".../minecraft-rl/", "relpaths": ["data/8.0/name.mp4", ...]}
  files:   <basedir>/data/8.0/name.mp4   (360p, 20 Hz)
           <basedir>/data/8.0/name.jsonl (one action dict per line)
Older documentation also shows relpaths without extensions; both forms are supported.
Some segments are missing from the bucket; those are skipped, not fatal.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from .encode import (canon_item, encode_state, mulaw_encode, resize_area, state_dim)

INDEX_BASE = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/"
DEFAULT_INDEXES = ["all_10xx_Jun_29.json", "all_7xx_Apr_6.json", "all_8xx_Jun_29.json"]

KEYMAP = {
    "key.keyboard.w": 0,
    "key.keyboard.s": 1,
    "key.keyboard.a": 2,
    "key.keyboard.d": 3,
    "key.keyboard.space": 4,
    "key.keyboard.left.shift": 5,
    "key.keyboard.left.control": 6,
}
ATTACK_BUTTON, USE_BUTTON = 0, 1
MOUSE_DEG_PER_PIXEL = 360.0 / 2400.0  # VPT's camera scaler; only used when pose is missing


# ----------------------------------------------------------------------------
# download helpers
# ----------------------------------------------------------------------------

def http_get(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> bool:
    """Resumable download. Returns False on 404 (missing segment), raises on repeated failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            start = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url)
            if start:
                req.add_header("Range", f"bytes={start}-")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (start and r.status == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(part, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):
                return False
            if e.code == 416 and part.exists():  # already complete
                os.replace(part, dest)
                return True
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after {retries} tries: {url}")


def fetch_index(name: str, cache_dir: Path) -> tuple[str, list[str]]:
    p = Path(cache_dir) / name
    if not p.exists():
        if not http_get(INDEX_BASE + name, p):
            raise RuntimeError(f"index not found: {name}")
    with open(p, "r", encoding="utf-8") as f:
        idx = json.load(f)
    base, paths = idx["basedir"], idx["relpaths"]
    if not isinstance(base, str) or not isinstance(paths, list) or not paths:
        raise ValueError(f"bad or empty VPT index: {p}")
    return base, paths


def pick_segments(index_names, n, cache_dir, seed=0):
    """Round-robin over indexes; n=None returns all candidates for missing-file fallback."""
    rng = random.Random(seed)
    pools = []
    for name in index_names:
        base, rel = fetch_index(name, cache_dir)
        rng.shuffle(rel)
        pools.append([(base, r) for r in rel])
    out, i = [], 0
    while (n is None or len(out) < n) and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            out.append(pool.pop())
        i += 1
    return out


def segment_stem(relpath: str) -> str:
    """Index paths can include .mp4; remove it BEFORE appending either extension."""
    relpath = relpath.strip().lstrip("/")
    if relpath.endswith((".mp4", ".jsonl")):
        return relpath.rsplit(".", 1)[0]
    return relpath


def segment_urls(basedir: str, relpath: str) -> tuple[str, str]:
    base = basedir.rstrip("/") + "/"
    stem = segment_stem(relpath)
    return (urllib.parse.urljoin(base, stem + ".mp4"),
            urllib.parse.urljoin(base, stem + ".jsonl"))


def seg_name(relpath: str) -> str:
    return segment_stem(relpath).replace("/", "__")


def download_segment(basedir: str, relpath: str, raw_dir: Path):
    name = seg_name(relpath)
    mp4 = Path(raw_dir) / f"{name}.mp4"
    jl = Path(raw_dir) / f"{name}.jsonl"
    video_url, action_url = segment_urls(basedir, relpath)
    if not ((jl.exists() and jl.stat().st_size > 0) or http_get(action_url, jl)):
        return None
    if not ((mp4.exists() and mp4.stat().st_size > 0) or http_get(video_url, mp4)):
        return None
    return mp4, jl


def download_first_valid(segments, raw_dir, n, workers=4, skip_names=(), max_attempts=80, log=print):
    """Try replacement index entries until n valid pairs exist, without over-downloading.

    The public index contains missing chunks. Only n jobs (including successes)
    are ever in flight, and failures are replaced with new candidates.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if n <= 0:
        return []
    skip = set(skip_names)
    candidates = []
    for base, rel in segments:
        name = seg_name(rel)
        if name not in skip:
            candidates.append((base, rel))
            skip.add(name)
        if len(candidates) >= max_attempts:
            break
    successes, attempted, index = [], 0, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = {}
        while len(successes) < n:
            while len(successes) + len(pending) < n and index < len(candidates):
                pair = candidates[index]
                pending[ex.submit(download_segment, *pair, raw_dir)] = pair
                index += 1
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pair = pending.pop(future)
                attempted += 1
                try:
                    if future.result() is not None:
                        successes.append(pair)
                        log(f"valid download {len(successes)}/{n}: {pair[1]}")
                    else:
                        log(f"missing in bucket: {pair[1]}")
                except Exception as exc:
                    log(f"download failed {pair[1]}: {type(exc).__name__}: {exc}")
    if len(successes) < n:
        raise RuntimeError(f"Only {len(successes)}/{n} VPT video/action pairs were available "
                           f"after {attempted} attempts. Check Colab internet access and index URLs.")
    return successes[:n]


def download_many(segments, raw_dir, workers=8, log=print):
    """Download .jsonl + .mp4 for many segments in parallel. Returns the (basedir, relpath) pairs that exist."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ok = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_segment, b, r, raw_dir): (b, r) for b, r in segments}
        for n, fut in enumerate(as_completed(futs), 1):
            seg = futs[fut]
            try:
                if fut.result() is not None:
                    ok.append(seg)
                else:
                    log(f"missing in bucket: {seg[1]}")
            except Exception as e:
                log(f"download failed {seg[1]}: {e}")
            if n % 10 == 0 or n == len(futs):
                log(f"downloaded {n}/{len(futs)}")
    order = {s: i for i, s in enumerate(segments)}
    return sorted(ok, key=lambda s: order[s])


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

def read_actions(jsonl_path) -> list[dict]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({})  # keep alignment with video frames
    return out


def inventory_counts(entry: dict) -> dict:
    inv = entry.get("inventory")
    counts: dict = {}
    if not inv:
        return counts
    items = inv.values() if isinstance(inv, dict) else inv
    for e in items:
        if not isinstance(e, dict):
            continue
        t = canon_item(str(e.get("type", "none")))
        if t == "none":
            continue
        q = e.get("quantity", 0)
        try:
            q = int(q)
        except (TypeError, ValueError):
            continue
        if q > 0:
            counts[t] = counts.get(t, 0) + q
    return counts


def build_vocab(jsonl_paths, k: int) -> list[str]:
    """Top-k items by total positive inventory change (what players actually collect and make)."""
    gain: Counter = Counter()
    for p in jsonl_paths:
        prev: dict | None = None
        for i, entry in enumerate(read_actions(p)):
            if i % 2:
                continue
            cur = inventory_counts(entry)
            if prev is not None:  # the first tick is the starting inventory, not a gain
                for name, c in cur.items():
                    d = c - prev.get(name, 0)
                    if d > 0:
                        gain[name] += d
            prev = cur
    return [n for n, _ in gain.most_common(k)]


def split_of(name: str) -> str:
    return "val" if zlib.crc32(name.encode()) % 10 == 0 else "train"


# ----------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------

def _wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def _ext(a, pad):
    return np.concatenate([a, np.repeat(a[-1:], pad)]) if len(a) else a


def parse_ticks(actions: list[dict]) -> dict:
    n = len(actions)
    keys = np.zeros((n, 9), dtype=bool)
    hotbar = np.full(n, -1, dtype=np.int64)
    yaw = np.full(n, np.nan)
    pitch = np.full(n, np.nan)
    gui = np.zeros(n, dtype=bool)
    dx = np.zeros(n)
    dy = np.zeros(n)
    has_inv = 0
    for i, a in enumerate(actions):
        kb = a.get("keyboard") or {}
        for k in kb.get("keys", []) or []:
            j = KEYMAP.get(k)
            if j is not None:
                keys[i, j] = True
        ms = a.get("mouse") or {}
        btn = ms.get("buttons", []) or []
        keys[i, 7] = ATTACK_BUTTON in btn
        keys[i, 8] = USE_BUTTON in btn
        dx[i] = float(ms.get("dx", 0.0) or 0.0)
        dy[i] = float(ms.get("dy", 0.0) or 0.0)
        if "hotbar" in a and a["hotbar"] is not None:
            hotbar[i] = int(a["hotbar"])
        if "yaw" in a and "pitch" in a:
            yaw[i], pitch[i] = float(a["yaw"]), float(a["pitch"])
        gui[i] = bool(a.get("isGuiOpen", False))
        has_inv += 1 if a.get("inventory") else 0
    return dict(n=n, keys=keys, hotbar=hotbar, yaw=yaw, pitch=pitch, gui=gui, dx=dx, dy=dy,
                has_pose=bool(np.isfinite(yaw).mean() > 0.5), has_hotbar=bool((hotbar >= 0).mean() > 0.5),
                has_inventory=bool(has_inv > n * 0.5))


def decode_frames(mp4_path, spec: dict, max_ticks: int):
    """Decode every tick but only resize the ones we keep. Returns (frames[N,H,W,3] uint8, ticks_decoded)."""
    import cv2

    stride, size = spec["stride"], spec["img_size"]
    cap = cv2.VideoCapture(str(mp4_path))
    frames, i = [], 0
    while i < max_ticks:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(resize_area(rgb, size, size))
        i += 1
    cap.release()
    return (np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)), i


def hindsight_goals(inv_steps: np.ndarray, horizon: int, rarity: np.ndarray) -> np.ndarray:
    """inv_steps: (N, K) counts. Goal = vocab item (1..K) gained most within the next `horizon` steps, 0 if none."""
    n, k = inv_steps.shape
    inc = np.zeros((n, k), dtype=np.float64)
    inc[1:] = np.maximum(inv_steps[1:] - inv_steps[:-1], 0)
    cs = np.cumsum(inc, axis=0)
    hi = np.minimum(np.arange(n) + horizon, n - 1)
    window = cs[hi] - cs                      # gains in (t, t+horizon]
    score = window + 1e-6 * rarity[None, :]
    best = score.argmax(axis=1)
    has = window.max(axis=1) > 0
    return np.where(has, best + 1, 0).astype(np.int64)


def convert_segment(mp4_path, jsonl_path, spec: dict, vocab: list[str], out_prefix: Path) -> dict:
    stride = spec["stride"]
    cam = spec["camera"]
    vocab_index = {n: i for i, n in enumerate(vocab)}
    k = spec["vocab_size"]

    actions = read_actions(jsonl_path)
    t = parse_ticks(actions)
    frames, decoded = decode_frames(mp4_path, spec, max_ticks=t["n"])
    n_use = min(decoded, t["n"])
    n_steps = min(n_use // stride, len(frames))
    if n_steps < 8:
        raise ValueError(f"too short after alignment ({n_steps} steps)")
    frames = frames[:n_steps]

    # --- labels over ticks [s*stride, s*stride+stride) -------------------------------
    tick = np.arange(n_steps) * stride
    keys = np.zeros((n_steps, 9), dtype=np.uint8)
    for o in range(stride):
        keys |= t["keys"][np.minimum(tick + o, t["n"] - 1)].astype(np.uint8)

    hot = _ext(t["hotbar"], stride)
    hot_now, hot_next = hot[tick], hot[tick + stride]
    hotbar_lab = np.where((hot_next >= 0) & (hot_next != hot_now), hot_next + 1, 0).astype(np.int64)
    hotbar_lab = np.clip(hotbar_lab, 0, spec["hotbar_classes"] - 1)

    if t["has_pose"]:
        yaw = np.nan_to_num(_ext(t["yaw"], stride), nan=0.0)
        pit = np.nan_to_num(_ext(t["pitch"], stride), nan=0.0)
        dyaw = _wrap180(yaw[tick + stride] - yaw[tick])
        dpit = pit[tick + stride] - pit[tick]
    else:
        dyaw = np.array([t["dx"][s * stride:(s + 1) * stride].sum() for s in range(n_steps)]) * MOUSE_DEG_PER_PIXEL
        dpit = np.array([t["dy"][s * stride:(s + 1) * stride].sum() for s in range(n_steps)]) * MOUSE_DEG_PER_PIXEL
        yaw = np.zeros(t["n"] + stride)
        pit = np.zeros(t["n"] + stride)
    cam_valid = (np.abs(dyaw) <= cam["invalid_deg"]) & (np.abs(dpit) <= cam["invalid_deg"])
    camy = mulaw_encode(dyaw, cam)
    camp = mulaw_encode(dpit, cam)

    gui_ext = _ext(t["gui"], stride)
    in_gui = gui_ext[tick] | gui_ext[tick + stride - 1]
    mask_all = (~in_gui).astype(np.uint8)

    # --- observation state + inventory-based goals ---------------------------------
    state = np.zeros((n_steps, state_dim(spec)), dtype=np.float32)
    inv_steps = np.zeros((n_steps, k), dtype=np.float64)
    for s in range(n_steps):
        inv = inventory_counts(actions[tick[s]]) if t["has_inventory"] else {}
        for name, c in inv.items():
            j = vocab_index.get(name)
            if j is not None:
                inv_steps[s, j] = c
        state[s] = encode_state(spec, int(hot_now[s]), float(pit[tick[s]]), float(yaw[tick[s]]),
                                bool(in_gui[s]), inv, vocab_index)
    rarity = -np.log1p(inv_steps.sum(axis=0) + 1.0)
    goal = hindsight_goals(inv_steps, spec["goal"]["horizon_steps"], rarity) if t["has_inventory"] \
        else np.zeros(n_steps, np.int64)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_frames.npy", frames)
    np.savez(str(out_prefix) + "_labels.npz",
             state=state, keys=keys, hotbar=hotbar_lab.astype(np.uint8),
             camy=camy.astype(np.uint8), camp=camp.astype(np.uint8),
             goal=goal.astype(np.uint8), mask_all=mask_all, mask_cam=cam_valid.astype(np.uint8))
    return dict(steps=int(n_steps), has_pose=t["has_pose"], has_hotbar=t["has_hotbar"],
                has_inventory=t["has_inventory"], gui_frac=float(in_gui.mean()),
                goal_frac=float((goal > 0).mean()), cam_valid_frac=float(cam_valid.mean()),
                frame_tick_gap=int(decoded - t["n"]))


def convert_many(segments, spec, vocab, raw_dir, shard_dir, delete_raw=True, log=print):
    """segments: list of (basedir, relpath). Returns list of per-segment stat dicts (resumable)."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    done_log = shard_dir / "done.jsonl"
    done = set()
    if done_log.exists():
        done = {json.loads(l)["name"] for l in done_log.read_text().splitlines() if l.strip()}
    stats = []
    for i, (base, rel) in enumerate(segments):
        name = seg_name(rel)
        if name in done:
            continue
        try:
            files = download_segment(base, rel, raw_dir)
            if files is None:
                log(f"[{i + 1}/{len(segments)}] missing in bucket, skipped: {rel}")
                continue
            mp4, jl = files
            st = convert_segment(mp4, jl, spec, vocab, shard_dir / name)
            st.update(name=name, split=split_of(name))
            with open(done_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(st) + "\n")
            stats.append(st)
            log(f"[{i + 1}/{len(segments)}] {name}: {st['steps']} steps, pose={st['has_pose']} "
                f"inv={st['has_inventory']} goal_frac={st['goal_frac']:.2f} gui={st['gui_frac']:.2f}")
            if delete_raw:
                mp4.unlink(missing_ok=True)
                jl.unlink(missing_ok=True)
        except Exception as e:  # keep going; one bad segment must not kill an hours-long run
            log(f"[{i + 1}/{len(segments)}] FAILED {rel}: {type(e).__name__}: {e}")
    return stats
