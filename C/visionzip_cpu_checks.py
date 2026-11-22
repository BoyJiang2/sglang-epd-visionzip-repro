import ast
import importlib.util
import itertools
import types
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import torch

torch.manual_seed(42)
root = Path(sys.argv[1]) / "python/sglang/srt"
spec = importlib.util.spec_from_file_location("visionzip", root/"disaggregation/encoder/visionzip.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

q = torch.randn(512 + 300, 4, 8)
k = torch.randn_like(q)
patch_counts = [512, 300]
reverse = torch.cat([torch.randperm(128), 128 + torch.randperm(75)])
masses, keys = module.image_statistics(q, k, patch_counts, 4, reverse)
ref_masses, ref_keys = [], []
for query, key in zip(q.split(patch_counts), k.split(patch_counts)):
    query, key = query.transpose(0, 1), key.transpose(0, 1)
    probabilities = ((query @ key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)).softmax(-1)
    ref_masses.append(probabilities.mean(0).sum(0).reshape(-1, 4).mean(-1))
    ref_keys.append(key.reshape(4, -1, 4, 8).mean(2).mean(0))
torch.testing.assert_close(masses, torch.cat(ref_masses)[reverse])
torch.testing.assert_close(keys, torch.cat(ref_keys)[reverse])

def author_reference(embedding, scores, key, ratio):
    peak = scores.topk(int((ratio - 5) * embedding.shape[0] / 100)).indices
    rest = [i for i in range(embedding.shape[0]) if i not in peak.tolist()]
    count = max(1, int(embedding.shape[0] * 0.05))
    anchor = rest[::max(1, len(rest) // count)][:count]
    assigned = defaultdict(list)
    for i in rest:
        if i not in anchor:
            sim = torch.nn.functional.cosine_similarity(key[i].unsqueeze(0), key[anchor])
            assigned[anchor[int(sim.argmax())]].append(i)
    keep = sorted(peak.tolist() + anchor)
    updated = embedding.clone()
    for pos, others in assigned.items():
        updated[pos] += embedding[others].mean(0)
    return updated[keep], keep

for ratio in (50, 70, 100):
    for n in (1, 15, 75, 128):
        embedding = torch.randn(n, 32)
        scores = torch.randn(n)
        key = torch.randn(n, 16)
        output, indices = module.compress_image(embedding, scores, key, ratio)
        ref, ref_indices = (
            (embedding, list(range(n))) if ratio == 100
            else author_reference(embedding, scores, key, ratio)
        )
        assert indices == ref_indices
        torch.testing.assert_close(output, ref)

original = [90, 91, 9, 9, 9, 9, 92, 93, 9, 9, 9, 94, 95]
positions = torch.arange(39).reshape(3, 13)
delta = torch.tensor([-2])
offsets = [(2, 5), (8, 10)]
keep = [[0, 2, 3], [1]]
cols = [0, 1, 2, 4, 5, 6, 7, 9, 11, 12]
ids, updated, p, d = module.compact_prompt(original, offsets, positions, delta, keep)
assert ids == [original[k] for k in cols]
assert updated == [(2, 4), (7, 7)]
assert torch.equal(p, positions[:, cols])
for decode_step in (0, 1, 10):
    assert len(ids) + decode_step + d == len(original) + decode_step + delta
c100 = module.compact_prompt(original, offsets, positions, delta, [[0, 1, 2, 3], [0, 1, 2]])
assert c100[0:2] == (original, offsets)
assert torch.equal(c100[2], positions) and torch.equal(c100[3], delta)
for invalid in ([[2, 0], [1]], [[0, 4], [1]], [[]], [[0, 0], [1]]):
    try:
        module.compact_prompt(original, offsets, positions, delta, invalid)
        raise AssertionError("malformed metadata accepted")
    except ValueError:
        pass

def run_source(relpath, names, ns, methods=None):
    node = ast.parse((root / relpath).read_text(encoding="utf-8"))
    source = [item for item in node.body if getattr(item,"name",None) in names or
              isinstance(item, ast.Assign) and any(getattr(t,"id",None) in names for t in item.targets)]
    if methods:
        cls = next(item for item in node.body if getattr(item,"name",None)==methods[0])
        source.extend(item for item in cls.body if getattr(item,"name",None) in methods[1])
    source.insert(0, ast.parse("from __future__ import annotations").body[0])
    exec(compile(ast.Module(body=source, type_ignores=[]), str(root / relpath), "exec"), ns)

class Modality(Enum):
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3

class InternalError(RuntimeError):
    pass

ns = dict(globals(), is_epd_accelerator=lambda device: False)
run_source("disaggregation/encoder/receiver.py", {
    "EmbeddingData", "MultiModalEmbeddingData", "_validate_embedding_part",
    "_MODALITY_GRID_ATTRS", "_GENERAL_IMAGE_META_ATTRS", "_GENERAL_VIDEO_META_ATTRS",
    "_MIMO_EXTRA_VIDEO_META_ATTRS", "_VIDEO_META_TENSOR_ATTRS", "_cat_grid",
    "video_meta_attrs_for",
}, ns)
EmbeddingData = ns["EmbeddingData"]
MultiModalEmbeddingData = ns["MultiModalEmbeddingData"]
run_source("disaggregation/encoder/server.py", set(), ns, ("MMEncoder", {"_stage_embeddings"}))
batch_features = torch.randn(7, 32)
ctx = types.SimpleNamespace(
    req_id="req-first", modality=Modality.IMAGE,
    items_per_req=[2, 1], use_global_cache=False,
    aux_data={"visionzip_indices":[[0, 2], [1], [0, 1, 2, 3]]},
    preprocess_result=types.SimpleNamespace(token_counts=[4, 3, 6],grid_thw=torch.tensor([[1,2,8],[1,2,6],[1,2,12]])),
)
requests = [
    {"req_id":"first","num_parts":1,"part_idx":0},
    {"req_id":"second","num_parts":1,"part_idx":0},
]
staged = []
fake_encoder = types.SimpleNamespace(rank=0, _stage_embedding_batch=staged.extend)
result = ns["_stage_embeddings"](fake_encoder, ctx, requests, batch_features, keep_on_gpu=False)
assert [r[1] for r in result] == [3, 4]
assert staged[0].visionzip_indices == [[0,2],[1]]
assert staged[1].visionzip_indices == [[0,1,2,3]]
torch.testing.assert_close(staged[0].get_embedding(), batch_features[:3])
torch.testing.assert_close(staged[1].get_embedding(), batch_features[3:])
first = EmbeddingData("example", 2, 0, torch.tensor([[1,2,8]]), Modality.IMAGE,
                      batch_features[:2], visionzip_indices=[[0,2]])
second = EmbeddingData("example", 2, 1, torch.tensor([[1,2,6]]), Modality.IMAGE,
                       batch_features[2:3], visionzip_indices=[[1]])
multi = MultiModalEmbeddingData.from_embedding_data(second)
multi.add(first)
assert multi.ready and multi.get_mm_extra_meta()["visionzip_indices"] == [[0,2],[1]]
torch.testing.assert_close(multi.get_embedding(is_concat=True)[Modality.IMAGE], batch_features[:3])

native_features = torch.randn(13, 32)
staged.clear()
ctx.aux_data.clear()
native = ns["_stage_embeddings"](
    fake_encoder, ctx, requests, native_features, keep_on_gpu=False
)
assert [r[1] for r in native] == [7, 6]
assert all(not hasattr(p, "visionzip_indices") for p in staged)
native_multi = MultiModalEmbeddingData.from_embedding_data(staged[0])
assert "visionzip_indices" not in native_multi.get_mm_extra_meta()
torch.testing.assert_close(
    native_multi.get_embedding(is_concat=True)[Modality.IMAGE], native_features[:7]
)
print(
    "PASS: attention mass parity, 50/70/100, malformed metadata, C100 and "
    "M-RoPE, batched staging, out-of-order parts, and native B staging."
)
