"""Write fixtures/parity.json from the Python reference so the Java core can be checked against it.

Run: python3 tests/gen_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from bbh.encode import (canon_item, encode_state, load_spec, mulaw_decode, mulaw_encode, resize_area)


def synth_image(h, w):
    """Deterministic image that both languages can build without shipping pixel data."""
    y = np.arange(h)[:, None, None]
    x = np.arange(w)[None, :, None]
    c = np.arange(3)[None, None, :]
    return ((y * 7 + x * 13 + c * 29 + ((y * x) % 17) * 5) % 256).astype(np.uint8)


def main():
    spec = load_spec()
    cam = spec["camera"]
    out = {"resize": [], "mulaw": {}, "canon": {}, "state": []}

    for (h, w) in [(360, 640), (768, 1366), (50, 50)]:
        img = synth_image(h, w)
        r = resize_area(img, spec["img_size"], spec["img_size"])
        out["resize"].append(dict(h=h, w=w, out=r.astype(int).flatten().tolist()))

    degs = [-100.0, -20.0, -19.99, -10.0, -3.3, -0.4, -0.01, 0.0, 0.01, 0.4, 1.0, 3.3, 7.5, 10.0, 19.99, 20.0, 55.0]
    out["mulaw"]["deg"] = degs
    out["mulaw"]["bin"] = mulaw_encode(degs, cam).astype(int).tolist()
    out["mulaw"]["decoded"] = mulaw_decode(np.arange(cam["bins"]), cam).tolist()

    names = ["minecraft:oak_log", "log", "log2", "dark_oak_log", "crimson_stem", "oak_planks", "Planks", "none", "air",
             "cobblestone", "stone_pickaxe", "white_wool", "spruce_sapling", "coarse_dirt", "iron_ore", "oak_leaves",
             "stripped_oak_wood", "minecraft:stone", "grass_block"]
    out["canon"] = {n: canon_item(n) for n in names}

    vocab = ["log", "planks", "cobblestone", "stone", "dirt"]
    vidx = {n: i for i, n in enumerate(vocab)}
    cases = [
        (0, 0.0, 0.0, False, {}),
        (4, -45.5, 135.25, True, {"log": 5, "planks": 64, "cobblestone": 1000, "unknown_item": 3}),
        (8, 90.0, -179.9, False, {"log": 1}),
    ]
    out["state_vocab"] = vocab
    for hb, pitch, yaw, gui, inv in cases:
        v = encode_state(spec, hb, pitch, yaw, gui, inv, vidx)
        out["state"].append(dict(hotbar=hb, pitch=pitch, yaw=yaw, gui=gui, inv=inv, out=[float(x) for x in v]))

    p = ROOT / "tests" / "fixtures"
    p.mkdir(exist_ok=True)
    (p / "parity.json").write_text(json.dumps(out))
    print("wrote", p / "parity.json", "-", (p / "parity.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
