"""Shared set encoder and a finite-context, noncausal residual TCN."""

import torch
from torch import nn


class SetEncoder(nn.Module):
    def __init__(self, width=64, output=128):
        super().__init__()
        self.player = nn.Sequential(
            nn.Linear(13, width), nn.GELU(), nn.Linear(width, width), nn.GELU()
        )
        self.attention = nn.Linear(width, 1)
        self.project = nn.Sequential(nn.Linear(width * 3 + 5, output), nn.GELU())

    def forward(self, xy, speed, ball, team, pitch):
        pm = torch.isfinite(xy).all(-1)
        bm = torch.isfinite(ball).all(-1)
        xy = torch.nan_to_num(xy)
        ball = torch.nan_to_num(ball)
        scale = pitch[:, None, None, :]
        centre = scale / 2
        sv = torch.isfinite(speed) & (speed >= 0) & (speed < 18) & pm
        speed = torch.nan_to_num(speed).clamp(0, 18)
        relative = (xy - ball[:, :, None, :]) * bm[:, :, None, None]
        distance = torch.linalg.vector_norm(relative, dim=-1) * bm[:, :, None]
        velocity = torch.zeros_like(xy)
        delta = (xy[:, 1:] - xy[:, :-1]) * 25
        vv = pm[:, 1:] & pm[:, :-1] & (torch.linalg.vector_norm(delta, dim=-1) < 13)
        velocity[:, 1:] = delta * vv[:, :, :, None]
        vvalid = torch.zeros_like(pm)
        vvalid[:, 1:] = vv
        # Opponent proximity uses instantaneous sets, never player identities.
        pair = torch.cdist(xy, xy)
        opponents = (
            (team[:, :, None] != team[:, None, :])
            & (team[:, :, None] >= 0)
            & (team[:, None, :] >= 0)
        )
        pair = pair.masked_fill(
            ~(opponents[:, None, :, :] & pm[:, :, :, None] & pm[:, :, None, :]), 1e4
        )
        enemy = pair.amin(-1)
        ev = enemy < 1e3
        enemy = enemy.masked_fill(~ev, 0)
        player_input = torch.cat(
            [
                (xy - centre) / scale,
                (speed / 10)[..., None],
                sv[..., None],
                relative / 50,
                (distance / 50)[..., None],
                bm[:, :, None, None].expand(-1, -1, xy.shape[2], -1),
                velocity / 10,
                vvalid[..., None],
                (enemy / 30)[..., None],
                ev[..., None],
            ],
            dim=-1,
        ).to(self.player[0].weight.dtype)
        encoded = self.player(player_input)
        out = []
        for side in (0, 1):
            mask = pm & (team[:, None, :] == side)
            count = mask.sum(-1, keepdim=True)
            denom = count.clamp_min(1)
            mean = (encoded * mask[..., None]).sum(-2) / denom
            maximum = encoded.masked_fill(~mask[..., None], -1e4).amax(-2)
            maximum = torch.where(count > 0, maximum, 0.0)
            attention = (
                self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1) * mask
            )
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            attended = (encoded * attention[..., None]).sum(-2)
            nearest = distance.masked_fill(~mask, 1e4).amin(-1, keepdim=True)
            nearest = torch.where((count > 0) & bm[..., None], nearest / 50, 0.0)
            average_speed = (speed * sv * mask).sum(-1, keepdim=True) / (
                (sv & mask).sum(-1, keepdim=True).clamp_min(1)
            )
            max_speed = speed.masked_fill(~(mask & sv), 0.0).amax(-1, keepdim=True)
            close = ((distance < 5) & mask & bm[:, :, None]).sum(-1, keepdim=True) / 11
            stats = torch.cat([count / 11, nearest, average_speed / 10, max_speed / 10, close], -1)
            out.append(self.project(torch.cat([mean, maximum, attended, stats], -1)))
        bv = torch.zeros_like(ball)
        valid = torch.zeros_like(bm)
        bd = (ball[:, 1:] - ball[:, :-1]) * 25
        valid[:, 1:] = bm[:, 1:] & bm[:, :-1] & (torch.linalg.vector_norm(bd, dim=-1) < 45)
        bv[:, 1:] = bd * valid[:, 1:, None]
        index = torch.arange(ball.shape[1], device=ball.device)[None, :].expand_as(bm)
        previous = torch.cummax(torch.where(bm, index, -ball.shape[1]), dim=1).values
        following = torch.cummin(
            torch.where(bm, index, ball.shape[1] * 2).flip(1), dim=1
        ).values.flip(1)
        age = ((index.to(ball.dtype) - previous) / 1500).clamp(0, 1)
        until = ((following - index.to(ball.dtype)) / 1500).clamp(0, 1)
        position = (ball - pitch[:, None, :] / 2) / pitch[:, None, :] * bm[..., None]
        global_features = torch.cat(
            [position, bm[..., None], bv / 30, valid[..., None], age[..., None], until[..., None]],
            -1,
        )
        return out[0], out[1], global_features.to(self.player[0].weight.dtype)


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.conv1(self.norm1(x.transpose(1, 2)).transpose(1, 2))
        z = self.dropout(self.activation(z))
        z = self.conv2(self.norm2(z.transpose(1, 2)).transpose(1, 2))
        return x + self.dropout(self.activation(z))


class PossessionTCN(nn.Module):
    def __init__(self, channels=128, blocks=8, set_width=64, set_output=128, dropout=0.1):
        super().__init__()
        self.config = dict(
            channels=channels,
            blocks=blocks,
            set_width=set_width,
            set_output=set_output,
            dropout=dropout,
        )
        self.set_encoder = SetEncoder(set_width, set_output)
        self.input = nn.Conv1d(set_output * 2 + 8, channels, 1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(channels, 2**i, dropout) for i in range(blocks)]
        )
        self.head = nn.Conv1d(channels, 11, 1)
        self.context = 2 * (2**blocks - 1)

    def _temporal(self, home, away, global_features):
        raw = self.head(
            self.blocks(self.input(torch.cat([home, away, global_features], -1).transpose(1, 2)))
        ).transpose(1, 2)
        dtype = self.head.weight.dtype
        dead = raw[..., 0].to(dtype).sigmoid()
        home_live = raw[..., 1].to(dtype).sigmoid()
        state = torch.stack([(1 - dead) * home_live, (1 - dead) * (1 - home_live), dead], -1)
        return state, raw[..., 2:].to(dtype).reshape(*raw.shape[:2], 3, 3)

    def forward(self, xy, speed, ball, team, pitch, symmetrise=True):
        home, away, global_features = self.set_encoder(xy, speed, ball, team, pitch)
        state, boundary = self._temporal(home, away, global_features)
        if symmetrise:
            other, other_boundary = self._temporal(away, home, global_features)
            order = [1, 0, 2]
            state = (state + other[..., order]) / 2
            boundary_prob = (
                boundary.sigmoid() + other_boundary[..., order, :][..., :, order].sigmoid()
            ) / 2
            boundary = torch.logit(boundary_prob.clamp(1e-6, 1 - 1e-6))
        return state.clamp_min(1e-7).log(), boundary
