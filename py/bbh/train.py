"""Goal-conditioned behaviour cloning on the shards written by data.convert_many."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encode import mulaw_decode
from .model import BBHPolicy, count_params


class ShardStore:
    def __init__(self, shard_dir, split: str, min_len: int = 64):
        shard_dir = Path(shard_dir)
        done = [json.loads(l) for l in (shard_dir / "done.jsonl").read_text().splitlines() if l.strip()]
        seen, self.segs = set(), []
        for d in done:
            if d["split"] != split or d["name"] in seen:
                continue
            seen.add(d["name"])
            fr = np.load(shard_dir / f"{d['name']}_frames.npy", mmap_mode="r")
            if len(fr) < min_len:   # too short to cut a training window from
                continue
            lb = dict(np.load(shard_dir / f"{d['name']}_labels.npz"))
            self.segs.append((fr, lb))
        if not self.segs:
            raise RuntimeError(f"no '{split}' segments (>= {min_len} steps) in {shard_dir}")
        self.lengths = np.array([len(fr) for fr, _ in self.segs], dtype=np.float64)
        self.p = self.lengths / self.lengths.sum()

    def total_steps(self):
        return int(self.lengths.sum())

    def batch(self, rng, batch_size, seq_len, goal_dropout, device):
        fr, st, ke, ho, cy, cp, go, ma, mc = ([] for _ in range(9))
        for _ in range(batch_size):
            i = rng.choice(len(self.segs), p=self.p)
            frames, lab = self.segs[i]   # every stored segment has >= min_len (64) steps
            s = rng.integers(0, len(frames) - seq_len + 1)
            sl = slice(s, s + seq_len)
            fr.append(np.asarray(frames[sl]))
            st.append(lab["state"][sl]); ke.append(lab["keys"][sl]); ho.append(lab["hotbar"][sl])
            cy.append(lab["camy"][sl]); cp.append(lab["camp"][sl])
            g = lab["goal"][sl].astype(np.int64)
            if rng.random() < goal_dropout:
                g = np.zeros_like(g)
            go.append(g); ma.append(lab["mask_all"][sl]); mc.append(lab["mask_cam"][sl])
        t = lambda a, dt: torch.from_numpy(np.stack(a)).to(device=device, dtype=dt)
        return dict(frames=torch.from_numpy(np.stack(fr)).to(device), state=t(st, torch.float32),
                    keys=t(ke, torch.float32), hotbar=t(ho, torch.long), cam_y=t(cy, torch.long),
                    cam_p=t(cp, torch.long), goal=t(go, torch.long),
                    mask=t(ma, torch.float32), mask_cam=t(mc, torch.float32))


HOTBAR_W = None


def compute_loss(out, b, burn, spec):
    global HOTBAR_W
    dev = out["keys"].device
    if HOTBAR_W is None or HOTBAR_W.device != dev:
        w = torch.ones(spec["hotbar_classes"]); w[0] = 0.02
        HOTBAR_W = w.to(dev)
    m = b["mask"].clone(); m[:, :burn] = 0
    mcam = m * b["mask_cam"]
    den = lambda x: x.sum().clamp(min=1.0)
    l_keys = (F.binary_cross_entropy_with_logits(out["keys"].float(), b["keys"], reduction="none").mean(-1) * m).sum() / den(m)
    l_hot = (F.cross_entropy(out["hotbar"].float().transpose(1, 2), b["hotbar"], weight=HOTBAR_W, reduction="none") * m).sum() / den(m)
    l_cy = (F.cross_entropy(out["cam_y"].float().transpose(1, 2), b["cam_y"], reduction="none") * mcam).sum() / den(mcam)
    l_cp = (F.cross_entropy(out["cam_p"].float().transpose(1, 2), b["cam_p"], reduction="none") * mcam).sum() / den(mcam)
    total = l_keys + 0.3 * l_hot + l_cy + l_cp
    return total, dict(loss=total.item(), keys=l_keys.item(), hotbar=l_hot.item(), cam_y=l_cy.item(), cam_p=l_cp.item())


@torch.no_grad()
def evaluate(model, store, spec, device, n_batches=30, seq_len=64, burn=8, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    cam = spec["camera"]
    tp = np.zeros(len(spec["keys"])); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    agg, n = {}, 0
    cam_hit = cam_tot = 0
    cam_err = []
    hot_hit = hot_tot = 0
    for _ in range(n_batches):
        b = store.batch(rng, 16, seq_len, 0.0, device)
        out, _ = model(b["frames"], b["state"], b["goal"])
        _, parts = compute_loss(out, b, burn, spec)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        m = b["mask"].clone(); m[:, :burn] = 0
        pred = (out["keys"] > 0).float(); y = b["keys"]
        mm = m.unsqueeze(-1)
        tp += ((pred * y) * mm).sum((0, 1)).cpu().numpy()
        fp += ((pred * (1 - y)) * mm).sum((0, 1)).cpu().numpy()
        fn += (((1 - pred) * y) * mm).sum((0, 1)).cpu().numpy()
        mc = m * b["mask_cam"]
        py, pp = out["cam_y"].argmax(-1), out["cam_p"].argmax(-1)
        cam_hit += (((py == b["cam_y"]) & (pp == b["cam_p"])).float() * mc).sum().item()
        cam_tot += mc.sum().item()
        dy = np.abs(mulaw_decode(py.cpu().numpy(), cam) - mulaw_decode(b["cam_y"].cpu().numpy(), cam))
        cam_err.append(float((dy * mc.cpu().numpy()).sum() / max(mc.sum().item(), 1)))
        ch = (b["hotbar"] > 0).float() * m
        hot_hit += (((out["hotbar"].argmax(-1) == b["hotbar"]).float()) * ch).sum().item()
        hot_tot += ch.sum().item()
    res = {k: v / n for k, v in agg.items()}
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    res["key_f1"] = {k: round(float(f), 3) for k, f in zip(spec["keys"], f1)}
    res["cam_joint_acc"] = cam_hit / max(cam_tot, 1)
    res["yaw_mae_deg"] = float(np.mean(cam_err))
    res["hotbar_change_recall"] = hot_hit / max(hot_tot, 1)
    model.train()
    return res


def run_training(shard_dir, out_dir, spec, steps=20000, batch_size=32, seq_len=32, burn=8, lr=3e-4,
                 warmup=200, eval_every=1000, save_every=500, max_minutes=None, seed=0, log=print):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_store = ShardStore(shard_dir, "train")
    try:
        val_store = ShardStore(shard_dir, "val")
    except RuntimeError:
        val_store = None
        log("no validation segments found; metrics will be training-only (convert more segments)")

    model = BBHPolicy(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_path = out_dir / "ckpt.pt"
    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck.get("history", [])
        log(f"resumed from step {step}")
    rng = np.random.default_rng(seed + step)
    log(f"device={device} params={count_params(model):,} train_steps_available={train_store.total_steps():,}")

    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, history=history, spec=spec), ckpt_path)

    t0 = time.time()
    model.train()
    running = {}
    running_steps = 0
    while step < steps:
        cur_lr = lr * min(1.0, (step + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        b = train_store.batch(rng, batch_size, seq_len, spec["goal"]["dropout"], device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out, _ = model(b["frames"], b["state"], b["goal"])
        loss, parts = compute_loss(out, b, burn, spec)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        running_steps += 1
        if step % 100 == 0:
            avg = {k: round(v / running_steps, 4) for k, v in running.items()}
            running = {}
            running_steps = 0
            history.append(dict(step=step, train=avg))
            log(f"step {step}/{steps} lr={cur_lr:.2e} {avg} ({(time.time() - t0) / 60:.1f} min)")
        if val_store is not None and step % eval_every == 0:
            ev = evaluate(model, val_store, spec, device)
            history.append(dict(step=step, val=ev))
            log(f"  VAL step {step}: loss={ev['loss']:.4f} cam_acc={ev['cam_joint_acc']:.3f} "
                f"yaw_mae={ev['yaw_mae_deg']:.2f}deg key_f1={ev['key_f1']}")
        if step % save_every == 0:
            save()
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            log(f"time budget reached at step {step}; saving. Re-run the cell to continue.")
            break
    if running_steps:  # a 30-step smoke test should still report its loss
        avg = {k: round(v / running_steps, 4) for k, v in running.items()}
        history.append(dict(step=step, train=avg))
        log(f"step {step}/{steps} {avg} ({(time.time() - t0) / 60:.1f} min)")
    save()
    return history
