"""Checkpoint -> single-step ONNX + the files the mod loads. Verifies ONNX against PyTorch."""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from .encode import state_dim
from .model import BBHPolicy, StepExport, count_params


def export_onnx(ckpt_path, out_dir, spec, vocab, goal_blocks_path=None, opset=17, log=print):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file
    policy = BBHPolicy(spec)
    policy.load_state_dict(ck["model"]); policy.eval()
    step_m = StepExport(policy).eval()

    s, d, h = spec["img_size"], state_dim(spec), spec["gru_hidden"]
    g = torch.Generator().manual_seed(0)
    frame = torch.randint(0, 256, (1, 3, s, s), generator=g).float()
    state = torch.randn(1, d, generator=g) * 0.2
    goal = torch.tensor([3], dtype=torch.long)
    h0 = torch.zeros(1, 1, h)
    path = out_dir / "policy.onnx"
    kwargs = dict(input_names=["frame", "state", "goal", "h"],
                  output_names=["keys", "hotbar", "cam_y", "cam_p", "h_out"],
                  opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(step_m, (frame, state, goal, h0), str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch has no dynamo kwarg
        torch.onnx.export(step_m, (frame, state, goal, h0), str(path), **kwargs)

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    # 30 sequential steps with the hidden state carried through, like the mod does
    max_diff, ht, ho = 0.0, h0.clone(), np.zeros((1, 1, h), np.float32)
    for i in range(30):
        fr = torch.randint(0, 256, (1, 3, s, s), generator=g).float()
        st = torch.randn(1, d, generator=g) * 0.2
        gl = torch.tensor([i % (spec["vocab_size"] + 1)], dtype=torch.long)
        with torch.no_grad():
            t_out = step_m(fr, st, gl, ht)
        o_out = sess.run(None, {"frame": fr.numpy(), "state": st.numpy(), "goal": gl.numpy(), "h": ho})
        for a, b_ in zip(t_out, o_out):
            max_diff = max(max_diff, float(np.abs(a.numpy() - b_).max()))
        ht, ho = t_out[4], o_out[4]
    assert max_diff < 1e-3, f"ONNX/PyTorch mismatch: {max_diff}"

    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {"frame": frame.numpy(), "state": state.numpy(), "goal": goal.numpy(), "h": h0.numpy()})
    ms = (time.perf_counter() - t0) / n * 1000

    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2))
    if goal_blocks_path and Path(goal_blocks_path).exists():
        shutil.copy(goal_blocks_path, out_dir / "goal_blocks.json")
    last_val = next((h["val"] for h in reversed(ck.get("history", [])) if "val" in h), None)
    meta = dict(params=count_params(policy), train_steps=ck["step"], onnx_max_abs_diff=max_diff,
                colab_cpu_ms_per_step_1thread=round(ms, 3), last_validation=last_val)
    (out_dir / "model_card.json").write_text(json.dumps(meta, indent=2))
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out_dir.iterdir():
            z.write(f, f"{out_dir.name}/{f.name}")
    log(f"exported {path} | params={meta['params']:,} | max diff vs torch={max_diff:.2e} | "
        f"{ms:.2f} ms/step on Colab CPU (1 thread) | zip: {zip_path}")
    return zip_path, meta
