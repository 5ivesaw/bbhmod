"""Small image + state + goal -> recurrent policy. Roughly 1M parameters, a few MFLOPs per step."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encode import state_dim


class BBHPolicy(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        size = spec["img_size"]
        s_dim = state_dim(spec)
        k = spec["vocab_size"]
        h = spec["gru_hidden"]
        bins = spec["camera"]["bins"]

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 3, size, size)).shape[1]
        self.img_fc = nn.Sequential(nn.Linear(flat, 128), nn.LayerNorm(128), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(s_dim, 128), nn.ReLU())
        self.goal_emb = nn.Embedding(k + 1, 32)
        self.fuse = nn.Sequential(nn.Linear(128 + 128 + 32, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(h, h), nn.ReLU())
        self.head_keys = nn.Linear(h, len(spec["keys"]))
        self.head_hotbar = nn.Linear(h, spec["hotbar_classes"])
        self.head_cam_y = nn.Linear(h, bins)
        self.head_cam_p = nn.Linear(h, bins)

    # frames_f: (N, 3, S, S) float in [0, 255]
    def encode(self, frames_f, state, goal):
        x = frames_f / 255.0 - 0.5
        img = self.img_fc(self.conv(x))
        return self.fuse(torch.cat([img, self.state_fc(state), self.goal_emb(goal)], dim=-1))

    def heads(self, t):
        return dict(keys=self.head_keys(t), hotbar=self.head_hotbar(t),
                    cam_y=self.head_cam_y(t), cam_p=self.head_cam_p(t))

    def forward(self, frames_u8, state, goal, h0=None):
        """frames_u8 (B,T,S,S,3) uint8, state (B,T,D) float, goal (B,T) long."""
        b, t = frames_u8.shape[:2]
        s = self.spec["img_size"]
        f = frames_u8.permute(0, 1, 4, 2, 3).reshape(b * t, 3, s, s).float()
        z = self.encode(f, state.reshape(b * t, -1), goal.reshape(b * t)).reshape(b, t, -1)
        out, h = self.gru(z, h0)
        return self.heads(self.trunk(out)), h


class StepExport(nn.Module):
    """Single-step graph used by the mod: frame (1,3,S,S) in 0..255, hidden state in and out."""

    def __init__(self, policy: BBHPolicy):
        super().__init__()
        self.p = policy

    def forward(self, frame, state, goal, h):
        z = self.p.encode(frame, state, goal).unsqueeze(1)
        out, h2 = self.p.gru(z, h)
        t = self.p.trunk(out[:, 0])
        o = self.p.heads(t)
        return o["keys"], o["hotbar"], o["cam_y"], o["cam_p"], h2


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
