"""Reference implementations of everything the mod must reproduce exactly.

The Java side (dev.bbh.agent.core) re-implements these functions and is checked
against fixtures generated from this file (tests/gen_fixtures.py). If you change
anything here, regenerate fixtures and re-run the Java parity test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------

def load_spec(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "contract" / "spec.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def state_dim(spec: dict) -> int:
    # hotbar one-hot + pitch + sin(yaw) + cos(yaw) + gui flag + inventory vocab
    return spec["state"]["hotbar_slots"] + 1 + 2 + 1 + spec["vocab_size"]


# ----------------------------------------------------------------------------
# item name canonicalisation (legacy VPT names and modern registry names meet here)
# ----------------------------------------------------------------------------

def canon_item(name: str) -> str:
    n = name.strip().lower()
    if ":" in n:
        n = n.split(":", 1)[1]
    if n in ("none", "air", "", "empty"):
        return "none"
    if n in ("log", "log2") or n.endswith("_log") or n.endswith("_stem") or n.endswith("_wood") \
            or n.endswith("_hyphae"):
        return "log"
    if n == "planks" or n.endswith("_planks"):
        return "planks"
    if n == "leaves" or n == "leaves2" or n.endswith("_leaves"):
        return "leaves"
    if n == "sapling" or n.endswith("_sapling"):
        return "sapling"
    if n == "dirt" or n in ("coarse_dirt", "rooted_dirt"):
        return "dirt"
    if n == "sand" or n == "red_sand":
        return "sand"
    if n == "stone" or n == "cobblestone":
        return n
    if n.endswith("_wool") or n == "wool":
        return "wool"
    return n


# ----------------------------------------------------------------------------
# camera quantisation (mu-law, same idea as VPT but 21 bins over +-max_deg)
# ----------------------------------------------------------------------------

def _round_half_up(v):
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5)


def mulaw_encode(deg, cam: dict):
    mx, mu, bins = cam["max_deg"], cam["mu"], cam["bins"]
    x = np.clip(np.asarray(deg, dtype=np.float64), -mx, mx) / mx
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / math.log1p(mu)
    idx = _round_half_up((y + 1.0) / 2.0 * (bins - 1))
    return np.clip(idx, 0, bins - 1).astype(np.int64)


def mulaw_decode(idx, cam: dict):
    mx, mu, bins = cam["max_deg"], cam["mu"], cam["bins"]
    y = np.asarray(idx, dtype=np.float64) / (bins - 1) * 2.0 - 1.0
    return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu * mx


# ----------------------------------------------------------------------------
# exact fractional-coverage area resize (identical algorithm in Java)
# ----------------------------------------------------------------------------

def area_weights(src: int, dst: int) -> np.ndarray:
    """(dst, src) matrix; row i holds the fractional coverage of source pixels by output cell i."""
    w = np.zeros((dst, src), dtype=np.float64)
    scale = src / dst
    for i in range(dst):
        lo = i * scale
        hi = (i + 1) * scale
        j0 = int(math.floor(lo))
        j1 = min(int(math.ceil(hi)), src)
        for j in range(j0, j1):
            cover = min(hi, j + 1) - max(lo, j)
            if cover > 0:
                w[i, j] = cover / scale
    return w


_W_CACHE: dict = {}


def resize_area(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """img: (H, W, 3) uint8 RGB -> (out_h, out_w, 3) uint8 RGB."""
    h, w = img.shape[:2]
    key = (h, out_h, w, out_w)
    if key not in _W_CACHE:
        _W_CACHE[key] = (
            area_weights(h, out_h).astype(np.float32),
            area_weights(w, out_w).astype(np.float32),
        )
    wy, wx = _W_CACHE[key]
    f = img.astype(np.float32)
    tmp = np.einsum("ow,hwc->hoc", wx, f)       # horizontal pass -> (H, out_w, 3)
    out = np.einsum("oh,hwc->owc", wy, tmp)     # vertical pass   -> (out_h, out_w, 3)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# state vector
# ----------------------------------------------------------------------------

def encode_state(spec: dict, hotbar: int, pitch: float, yaw: float, gui_open: bool,
                 inventory: dict, vocab_index: dict) -> np.ndarray:
    """inventory: canonical item name -> count. vocab_index: canonical name -> 0..K-1."""
    k = spec["vocab_size"]
    slots = spec["state"]["hotbar_slots"]
    cap = spec["state"]["inventory_count_cap"]
    out = np.zeros(state_dim(spec), dtype=np.float32)
    if 0 <= hotbar < slots:
        out[hotbar] = 1.0
    o = slots
    out[o] = pitch / 90.0
    rad = math.radians(yaw)
    out[o + 1] = math.sin(rad)
    out[o + 2] = math.cos(rad)
    out[o + 3] = 1.0 if gui_open else 0.0
    base = o + 4
    denom = math.log(1.0 + cap)
    for name, count in inventory.items():
        j = vocab_index.get(name)
        if j is not None and count > 0:
            out[base + j] = math.log(1.0 + min(count, cap)) / denom
    return out
