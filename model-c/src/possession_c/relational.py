"""Relational set encoder with an explicit ball-observation context."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from possession_c.model import PossessionTCN, SetEncoder
from possession_c.windows import collate, extract_window
from possession_c.schema import period_runs

CAP = 1500
MARGIN = 544
CENTRAL = 1024
LENGTH = CENTRAL + 2 * MARGIN


def ball_context(ball, cap=CAP):
    valid = np.isfinite(ball).all(-1)
    index = np.arange(len(ball))
    previous = np.maximum.accumulate(np.where(valid, index, -cap))
    following = np.minimum.accumulate(np.where(valid, index, len(ball) + cap)[::-1])[::-1]
    return (
        np.stack(
            [np.clip(index - previous, 0, cap), np.clip(following - index, 0, cap)], -1
        ).astype("f4")
        / cap
    )


def crop_context_sample(sample, length=LENGTH):
    context = ball_context(sample["ball"])
    result = dict(sample)
    for key in ("xy", "speed", "ball"):
        if key in result:
            result[key] = result[key][CAP : CAP + length].copy()
    result["ball_context"] = context[CAP : CAP + length].copy()
    return result


def set_velocity(xy, valid, team, lag=5):
    """Soft geometric correspondence to the preceding same-team set plus null.

    Null mass measures geometric support, not identity uniqueness. This is equivariant to
    independent within-team permutations at every time, including missing slots.
    """
    velocity = torch.zeros_like(xy)
    confidence = torch.zeros_like(valid, dtype=xy.dtype)
    if xy.shape[1] <= lag:
        return velocity, confidence
    now, before = xy[:, lag:], xy[:, :-lag]
    physics_dtype = torch.float64 if xy.dtype == torch.float64 else torch.float32
    distance = torch.cdist(now.to(physics_dtype), before.to(physics_dtype)).square()
    same = (team[:, :, None] == team[:, None, :]) & (team[:, :, None] >= 0)
    mask = same[:, None] & valid[:, lag:, :, None] & valid[:, :-lag, None, :]
    score = (-distance / 2.0).masked_fill(~mask, -1e4)
    null = torch.full_like(score[..., :1], -2.0)
    weights = torch.cat([score, null], -1).softmax(-1)[..., :-1] * mask
    mass = weights.sum(-1)
    position = torch.matmul(weights, before.to(weights.dtype)) / mass[..., None].clamp_min(1e-8)
    motion = (now - position) * (25.0 / lag)
    reliable = (torch.linalg.vector_norm(motion, dim=-1) < 13) & valid[:, lag:]
    confidence[:, lag:] = mass * reliable
    velocity[:, lag:] = motion * confidence[:, lag:, :, None]
    return velocity, confidence


class RelationBlock(nn.Module):
    def __init__(self, width, heads=2, compact=False):
        super().__init__()
        self.heads, self.width = heads, width
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.bias = (
            nn.Linear(4, heads)
            if compact
            else nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, heads))
        )
        self.project = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, encoded, xy, team, valid):
        b, t, n, d = encoded.shape
        qkv = self.qkv(self.norm(encoded)).reshape(b, t, n, 3, self.heads, d // self.heads)
        q, k, v = [qkv[:, :, :, i].permute(0, 1, 3, 2, 4) for i in range(3)]
        relative = (xy[:, :, :, None, :] - xy[:, :, None, :, :]) / 10
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        same = (team[:, :, None] == team[:, None, :])[:, None, :, :, None].expand(b, t, n, n, 1)
        bias = self.bias(torch.cat([relative, distance, same.to(relative.dtype)], -1)).permute(
            0, 1, 4, 2, 3
        )
        scores = (q @ k.transpose(-1, -2)) / (d // self.heads) ** 0.5 + bias
        mask = valid[:, :, None, None, :]
        weights = scores.to(self.qkv.weight.dtype).masked_fill(~mask, -1e4).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        result = (weights.to(v.dtype) @ v).permute(0, 1, 3, 2, 4).reshape(b, t, n, d)
        out = encoded + self.project(result)
        return (out + self.ff(out)) * valid[..., None]


class RelationalEncoder(SetEncoder):
    def __init__(self, width=64, output=128, compact=False):
        super().__init__(width, output)
        self.ball_token = nn.Linear(3, width)
        self.relations = nn.ModuleList(
            [RelationBlock(width, compact=compact), RelationBlock(width, compact=compact)]
        )

    def forward(self, xy, speed, ball, team, pitch):
        pm = torch.isfinite(xy).all(-1) & (team[:, None] >= 0)
        bm = torch.isfinite(ball).all(-1)
        xy = torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.)
        ball = torch.nan_to_num(ball, nan=0., posinf=0., neginf=0.)
        scale = pitch[:, None, None, :]
        sv = torch.isfinite(speed) & (speed >= 0) & (speed < 18) & pm
        speed = torch.nan_to_num(speed, nan=0., posinf=0., neginf=0.).clamp(0, 18)
        relative = (xy - ball[:, :, None, :]) * bm[:, :, None, None]
        distance = torch.linalg.vector_norm(relative, dim=-1) * bm[:, :, None]
        velocity, confidence = set_velocity(xy, pm, team)
        physics_dtype = torch.float64 if xy.dtype == torch.float64 else torch.float32
        pair = torch.cdist(xy.to(physics_dtype), xy.to(physics_dtype))
        opponents = (
            (team[:, :, None] != team[:, None, :])
            & (team[:, :, None] >= 0)
            & (team[:, None, :] >= 0)
        )
        pair = pair.masked_fill(~(opponents[:, None] & pm[:, :, :, None] & pm[:, :, None, :]), 1e4)
        enemy = pair.amin(-1)
        ev = enemy < 1e3
        enemy = enemy.masked_fill(~ev, 0)
        features = torch.cat(
            [
                (xy - scale / 2) / scale,
                (speed / 10)[..., None],
                sv[..., None],
                relative / 50,
                (distance / 50)[..., None],
                bm[:, :, None, None].expand(-1, -1, xy.shape[2], -1),
                velocity / 10,
                confidence[..., None],
                (enemy / 30)[..., None],
                ev[..., None],
            ],
            -1,
        )
        encoded = self.player(features.to(self.player[0].weight.dtype))
        position = (ball - pitch[:, None, :] / 2) / pitch[:, None, :] * bm[..., None]
        token = self.ball_token(torch.cat([position, bm[..., None]], -1).to(encoded.dtype))
        encoded = torch.cat([encoded, token[:, :, None]], -2)
        all_xy = torch.cat([xy, ball[:, :, None]], -2)
        all_team = torch.cat([team, torch.full_like(team[:, :1], 2)], -1)
        all_valid = torch.cat([pm, bm[..., None]], -1)
        for layer in self.relations:
            if self.training and torch.is_grad_enabled() and getattr(self,'checkpoint_relations',False):
                from torch.utils.checkpoint import checkpoint
                encoded=checkpoint(layer,encoded,all_xy,all_team,all_valid,use_reentrant=False)
            else:
                encoded = layer(encoded, all_xy, all_team, all_valid)
        encoded = encoded[:, :, :-1]
        out = []
        for side in (0, 1):
            mask = pm & (team[:, None, :] == side)
            count = mask.sum(-1, keepdim=True)
            mean = (encoded * mask[..., None]).sum(-2) / count.clamp_min(1)
            maximum = encoded.masked_fill(~mask[..., None], -1e4).amax(-2)
            maximum = torch.where(count > 0, maximum, 0.0)
            attention = (
                self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1) * mask
            )
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            attended = (encoded * attention[..., None]).sum(-2)
            nearest = distance.masked_fill(~mask, 1e4).amin(-1, keepdim=True)
            nearest = torch.where((count > 0) & bm[..., None], nearest / 50, 0.0)
            average = (speed * sv * mask).sum(-1, keepdim=True) / (sv & mask).sum(
                -1, keepdim=True
            ).clamp_min(1)
            maximum_speed = speed.masked_fill(~(mask & sv), 0.0).amax(-1, keepdim=True)
            close = ((distance < 5) & mask & bm[:, :, None]).sum(-1, keepdim=True) / 11
            stats = torch.cat([count / 11, nearest, average / 10, maximum_speed / 10, close], -1)
            out.append(self.project(torch.cat([mean, maximum, attended, stats], -1)))
        bv = torch.zeros_like(ball)
        valid = torch.zeros_like(bm)
        delta = (ball[:, 1:] - ball[:, :-1]) * 25
        valid[:, 1:] = bm[:, 1:] & bm[:, :-1] & (torch.linalg.vector_norm(delta, dim=-1) < 45)
        bv[:, 1:] = delta * valid[:, 1:, None]
        # The explicit external context replaces these last two fields.
        global_features = torch.cat(
            [position, bm[..., None], bv / 30, valid[..., None], torch.ones_like(ball)], -1
        )
        return out[0], out[1], global_features.to(self.player[0].weight.dtype)


class RelationalPossession(PossessionTCN):
    def __init__(self, channels=128, blocks=8, set_width=64, set_output=128, dropout=0.1,
                 num_classes=3, compact=True):
        super().__init__(channels, blocks, set_width, set_output, dropout, num_classes)
        self.config["compact"] = compact
        self.set_encoder = RelationalEncoder(set_width, set_output, compact=compact)
        self.context += 5

    def _raw(self, home, away, global_features):
        initial = self.input(torch.cat([home, away, global_features], -1).transpose(1, 2))
        features = self.blocks(initial)
        raw = self.head(features).transpose(1, 2).to(self.head.weight.dtype)
        if self.num_classes == 2:
            state = torch.stack([F.logsigmoid(raw[..., 0]), F.logsigmoid(-raw[..., 0])], -1)
            return state, raw[..., 1:].reshape(*raw.shape[:2], 2, 2)
        state = torch.stack(
            [
                F.logsigmoid(-raw[..., 0]) + F.logsigmoid(raw[..., 1]),
                F.logsigmoid(-raw[..., 0]) + F.logsigmoid(-raw[..., 1]),
                F.logsigmoid(raw[..., 0]),
            ],
            -1,
        )
        return state, raw[..., 2:].reshape(*raw.shape[:2], 3, 3)

    def forward(self, xy, speed, ball, team, pitch, ball_context, symmetrise=True):
        home, away, glob = self.set_encoder(xy, speed, ball, team, pitch)
        glob = torch.cat([glob[..., :6], ball_context.to(glob.dtype)], -1)
        logp, boundary = self._raw(home, away, glob)
        if symmetrise:
            other, other_boundary = self._raw(away, home, glob)
            order = [1, 0] if self.num_classes == 2 else [1, 0, 2]
            logp = torch.logaddexp(logp, other[..., order]) - np.log(2.0)
            probability = (
                boundary.sigmoid() + other_boundary[..., order, :][..., :, order].sigmoid()
            ) / 2
            boundary = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return logp, boundary


def collate_context(samples):
    result = collate(samples)
    result["ball_context"] = torch.from_numpy(
        np.stack([sample["ball_context"] for sample in samples])
    )
    return result


@torch.inference_mode()
def predict_relational(model, sequence, device="cpu", batch_size=4, central=CENTRAL):
    model.eval()
    margin = max(MARGIN, model.context)
    length = central + 2 * margin
    dtype = next(model.parameters()).dtype
    numpy_dtype = "f8" if dtype == torch.float64 else "f4"
    classes = model.num_classes
    state = np.zeros((len(sequence.period), classes), numpy_dtype)
    boundary = np.zeros((len(sequence.period), classes, classes), numpy_dtype)
    tasks = [
        (run, start, min(central, run[1] - start))
        for run in period_runs(sequence.period)
        for start in range(run[0], run[1], central)
    ]
    for index in range(0, len(tasks), batch_size):
        part = tasks[index : index + batch_size]
        samples = [
            crop_context_sample(
                extract_window(sequence, start - margin - CAP, length + 2 * CAP, run), length
            )
            for run, start, n in part
        ]
        batch = collate_context(samples)
        batch = {
            k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()
        }
        logp, edge = model(**batch)
        for i, (run, start, n) in enumerate(part):
            state[start : start + n] = logp[i, margin : margin + n].cpu().numpy()
            boundary[start : start + n] = edge[i, margin : margin + n].cpu().numpy()
    return state, boundary
