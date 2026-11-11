"""End-to-end test of the converter on a synthetic segment written in the documented VPT format.

Run:  PYTHONPATH=py python3 tests/test_convert.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
from bbh import data
from bbh.encode import load_spec, mulaw_decode, state_dim


def make_segment(d: Path, n=600):
    rng = np.random.default_rng(1)
    lines = []
    yaw, pitch = 10.0, 5.0
    inv_log = 0
    for i in range(n):
        dyaw = 3.0 if 100 <= i < 140 else 0.0          # turn right for 2 s
        yaw += dyaw
        pitch += 0.5 if 300 <= i < 310 else 0.0
        if i == 200:
            inv_log = 3                                 # collect 3 logs at tick 200
        keys = ["key.keyboard.w"] if i < 300 else []
        if 50 <= i < 60:
            keys.append("key.keyboard.space")
        buttons = [0] if 150 <= i < 200 else []
        gui = 400 <= i < 440
        inv = [{"type": "none", "quantity": 0}] * 5
        if inv_log:
            inv = [{"type": "log", "quantity": inv_log}] + inv
        lines.append(json.dumps(dict(
            mouse=dict(x=0, y=0, dx=dyaw / data.MOUSE_DEG_PER_PIXEL, dy=0, dwheel=0, buttons=buttons, newButtons=[]),
            keyboard=dict(keys=keys, newKeys=[], chars=""),
            isGuiOpen=gui, isGuiInventory=gui, hotbar=0 if i < 500 else 2,
            yaw=((yaw + 180) % 360) - 180, pitch=pitch, xpos=i * 0.1, ypos=64.0, zpos=0.0,
            tick=i, milli=i * 50, inventory=inv, stats={})))
    (d / "seg.jsonl").write_text("\n".join(lines) + "\n")
    vw = cv2.VideoWriter(str(d / "seg.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360))
    for i in range(n):
        f = np.zeros((360, 640, 3), np.uint8)
        f[:, :, 2] = 255 if i % 40 < 20 else 0          # red channel flickers (BGR)
        f[:, : i % 640, 1] = 200
        vw.write(f)
    vw.release()


def main():
    spec = load_spec()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        make_segment(d)

        vocab = data.build_vocab([d / "seg.jsonl"], spec["vocab_size"])
        assert vocab == ["log"], vocab

        st = data.convert_segment(d / "seg.mp4", d / "seg.jsonl", spec, vocab, d / "out" / "seg")
        fr = np.load(d / "out" / "seg_frames.npy")
        lb = np.load(d / "out" / "seg_labels.npz")
        n = st["steps"]
        assert fr.shape == (n, 64, 64, 3) and fr.dtype == np.uint8, fr.shape
        assert abs(n - 300) <= 1, n                       # 600 ticks / stride 2
        assert lb["state"].shape == (n, state_dim(spec))
        assert st["has_pose"] and st["has_inventory"] and st["has_hotbar"]

        # camera: 3 deg/tick for 40 ticks -> 6 deg per step during steps 50..69
        decoded = mulaw_decode(lb["camy"], spec["camera"])
        mid = decoded[55:65]
        assert np.all(np.abs(mid - 6.0) < 1.5), mid       # bin quantisation error allowed
        assert np.all(lb["camy"][:45] == spec["camera"]["bins"] // 2)
        # pitch: 0.5 deg/tick for 10 ticks (ticks 300..309) -> steps 150..154 at ~1 deg
        assert decoded[0] == 0

        # keys: w held for first 300 ticks = first 150 steps; space at ticks 50..59 -> steps 25..29
        assert lb["keys"][10, 0] == 1 and lb["keys"][200, 0] == 0
        assert lb["keys"][27, 4] == 1 and lb["keys"][40, 4] == 0
        assert lb["keys"][80, 7] == 1 and lb["keys"][120, 7] == 0       # attack ticks 150..199 -> steps 75..99

        # gui: ticks 400..439 -> steps 200..219 masked out
        assert lb["mask_all"][210] == 0 and lb["mask_all"][190] == 1 and lb["mask_all"][230] == 1

        # hotbar switch at tick 500 -> slot 2 -> class 3 around step 249/250
        assert (lb["hotbar"] == 3).sum() == 1, np.nonzero(lb["hotbar"])

        # goal: logs gained at tick 200 (step 100). horizon is 150 steps -> all steps 0..99 have goal log (=1)
        assert np.all(lb["goal"][:100] == 1), lb["goal"][:100]
        assert np.all(lb["goal"][101:] == 0)

        # frames: red channel flickers; RGB mean of channel 0 must alternate high/low
        r = fr[:, :, :, 0].mean(axis=(1, 2))
        assert r.max() > 200 and r.min() < 5, (r.min(), r.max())

        # convert_many resume + skip logic
        stats = data.convert_many([("file://", "x/y")], spec, vocab, d / "raw", d / "shards", log=lambda *_: None)
        assert stats == []                                 # unreachable -> skipped/failed, not crashed
    print("convert tests passed")


if __name__ == "__main__":
    main()
