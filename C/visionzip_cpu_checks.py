import ast
import asyncio
import importlib.util
import itertools
import types
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import torch

torch.manual_seed(42)
root = Path(sys.argv[1]) / "python/sglang/srt"
native_root = Path(sys.argv[2]) / "python/sglang/srt" if len(sys.argv) > 2 else root
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

def run_source(relpath, names, ns, methods=None, *, source_root=root):
    path = source_root / relpath
    node = ast.parse(path.read_text(encoding="utf-8"))
    source = [item for item in node.body if getattr(item,"name",None) in names or
              isinstance(item, ast.Assign) and any(getattr(t,"id",None) in names for t in item.targets)]
    if methods:
        cls = next(item for item in node.body if getattr(item,"name",None)==methods[0])
        source.extend(item for item in cls.body if getattr(item,"name",None) in methods[1])
    source.insert(0, ast.parse("from __future__ import annotations").body[0])
    exec(compile(ast.Module(body=source, type_ignores=[]), str(path), "exec"), ns)

class Modality(Enum):
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3

class InternalError(RuntimeError):
    pass

ns = dict(globals(), is_epd_accelerator=lambda device: False)
ns["image_token_budget"] = getattr(module, "image_token_budget", None)
run_source("disaggregation/encoder/receiver.py", {
    "EmbeddingData", "MultiModalEmbeddingData", "_validate_embedding_part",
    "_MODALITY_GRID_ATTRS", "_GENERAL_IMAGE_META_ATTRS", "_GENERAL_VIDEO_META_ATTRS",
    "_MIMO_EXTRA_VIDEO_META_ATTRS", "_VIDEO_META_TENSOR_ATTRS", "_cat_grid",
    "video_meta_attrs_for",
}, ns)
EmbeddingData = ns["EmbeddingData"]
MultiModalEmbeddingData = ns["MultiModalEmbeddingData"]
run_source(
    "disaggregation/encoder/server.py", set(), ns,
    ("MMEncoder", {
        "_stage_embeddings", "_publish_preprocess_metadata",
        "_embedding_state_for_stage", "_stage_embedding_batch",
    }),
)
batch_features = torch.randn(7, 32)
ctx = types.SimpleNamespace(
    req_id="req-first", modality=Modality.IMAGE,
    items_per_req=[2, 1], use_global_cache=False,
    aux_data={"visionzip_indices":[[0, 2], [0], [0, 1, 2, 3]]},
    preprocess_result=types.SimpleNamespace(token_counts=[3, 1, 6],grid_thw=torch.tensor([[1,2,6],[1,2,2],[1,2,12]])),
)
requests = [
    {"req_id":"first","num_parts":1,"part_idx":0},
    {"req_id":"second","num_parts":1,"part_idx":0},
]
staged = []
fake_encoder = types.SimpleNamespace(
    rank=0, visionzip_ratio=70, _stage_embedding_batch=staged.extend,
)
result = ns["_stage_embeddings"](fake_encoder, ctx, requests, batch_features, keep_on_gpu=False)
assert [r[1] for r in result] == [3, 4]
assert staged[0].visionzip_indices == [[0,2],[0]]
assert staged[1].visionzip_indices == [[0,1,2,3]]
torch.testing.assert_close(staged[0].get_embedding(), batch_features[:3])
torch.testing.assert_close(staged[1].get_embedding(), batch_features[3:])


async def check_encoder_metadata():
    published = []

    async def publish(*args):
        published.append(args)

    ns["meta_registry"] = types.SimpleNamespace(publish=publish)
    request = requests[0]
    for ratio, health_check, rows in (
        (0, False, 2691), (100, False, 2691), (70, False, 1883),
        (50, False, 1344), (70, True, 2691),
    ):
        compressed_ctx = types.SimpleNamespace(
            req_id=request["req_id"], modality=Modality.IMAGE,
            is_health_check=health_check, items_per_req=[1], use_global_cache=False,
            aux_data={"visionzip_indices": [list(range(rows))]}
            if ratio and not health_check else {},
            preprocess_result=types.SimpleNamespace(
                token_counts=[2691], grid_thw=torch.tensor([[1, 78, 138]]),
            ),
        )
        state = types.SimpleNamespace(embedding_data=None, embedding_ready=asyncio.Event())
        encoder = types.SimpleNamespace(
            rank=0, visionzip_ratio=ratio, _embedding_dims={Modality.IMAGE: 3584},
            _embedding_dtype=torch.bfloat16, _element_size=2,
            _require_active_encode_state=lambda req_id: state,
            _deregister_shared_mr=lambda _: None,
        )
        for name in ("_embedding_state_for_stage", "_stage_embedding_batch"):
            setattr(encoder, name, types.MethodType(ns[name], encoder))
        await ns["_publish_preprocess_metadata"](encoder, compressed_ctx, [request])
        assert published[-1] == (request["req_id"], rows * 3584 * 2, rows, 3584), (
            ratio, health_check, published[-1]
        )
        if ratio == 70 and not health_check:
            wrong_dtype = torch.empty(rows, 3584, dtype=torch.float16)
            try:
                ns["_stage_embeddings"](
                    encoder, compressed_ctx, [request], wrong_dtype, keep_on_gpu=False,
                )
            except InternalError as error:
                assert "Embedding metadata mismatch" in str(error)
            else:
                raise AssertionError("Encoder staged embeddings with the wrong dtype")
            assert not state.embedding_ready.is_set()
        features = torch.empty(rows, 3584, dtype=torch.bfloat16)
        result = ns["_stage_embeddings"](
            encoder, compressed_ctx, [request], features, keep_on_gpu=False,
        )
        assert result == [(features.nbytes, rows, 3584, None, None)]
        assert state.embedding_ready.is_set()


asyncio.run(check_encoder_metadata())

first = EmbeddingData("example", 2, 0, torch.tensor([[1,2,8]]), Modality.IMAGE,
                      batch_features[:2], visionzip_indices=[[0,2]])
second = EmbeddingData("example", 2, 1, torch.tensor([[1,2,6]]), Modality.IMAGE,
                       batch_features[2:3], visionzip_indices=[[1]])
multi = MultiModalEmbeddingData.from_embedding_data(second)
multi.add(first)
assert multi.ready and multi.get_mm_extra_meta()["visionzip_indices"] == [[0,2],[1]]
torch.testing.assert_close(multi.get_embedding(is_concat=True)[Modality.IMAGE], batch_features[:3])

def load_position_pipeline(source_root):
    scope = dict(
        torch=torch,
        Modality=Modality,
        compact_prompt=module.compact_prompt,
        MultimodalDataItem=types.SimpleNamespace,
        MultimodalProcessorOutput=types.SimpleNamespace,
    )
    for relpath, cls, method in (
        ("multimodal/processors/base_processor.py", "BaseMultimodalProcessor", "build_input_ids"),
        ("multimodal/processors/qwen_vl.py", "QwenVLImageProcessor", "get_mm_data"),
        ("layers/rotary_embedding/mrope.py", "MRotaryEmbedding", "get_rope_index"),
        ("model_executor/forward_batch_info.py", "ForwardBatch", "_expand_mrope_from_input"),
    ):
        run_source(relpath, set(), scope, (cls, {method}), source_root=source_root)
    rope_name = "sglang.srt.multimodal.mrope_rope_index"
    rope_spec = importlib.util.spec_from_file_location(
        rope_name, source_root / "multimodal/mrope_rope_index.py"
    )
    rope_module = importlib.util.module_from_spec(rope_spec)
    rope_spec.loader.exec_module(rope_module)

    def get_rope_index(**kwargs):
        with patch.dict(sys.modules, {rope_name: rope_module}):
            return scope["get_rope_index"](**kwargs)

    scope["MRotaryEmbedding"] = types.SimpleNamespace(get_rope_index=get_rope_index)
    return scope


def assemble(scope, prompt, grid, features, **kwargs):
    processor = types.SimpleNamespace(
        model_type="qwen2_5_vl",
        _spatial_merge_size=2,
        spatial_merge_size=2,
        mm_tokens=types.SimpleNamespace(
            image_token_id=151655, video_token_id=151656, audio_token_id=None
        ),
        IM_TOKEN_ID=151655,
        VIDEO_TOKEN_ID=151656,
        vision_start_token_id=151652,
        audio_start_token_id=None,
        IM_START_TOKEN_ID=151652,
        IM_END_TOKEN_ID=151653,
        hf_config=types.SimpleNamespace(),
        _tokens_per_second=2,
    )
    processor.build_input_ids = types.MethodType(scope["build_input_ids"], processor)
    output = scope["get_mm_data"](
        processor, prompt, {Modality.IMAGE: features}, img_grid_thw=grid, **kwargs
    )
    output.mrope_position_delta_repeated_cache = None
    return output


native_pipeline = load_position_pipeline(native_root)
compressed_pipeline = load_position_pipeline(root)
grids = [
    (1, 6, 10), (1, 10, 6), (1, 4, 8), (1, 8, 14),
    (1, 14, 6), (1, 4, 12), (1, 8, 4), (1, 78, 138),
]
for image_count in (1, 4, 8):
    grid = torch.tensor(grids[:image_count])
    prompt = [100]
    for i in range(image_count):
        prompt.extend([101 + i, 151652, 151655, 151653])
    prompt.extend([110, 111])
    lengths = (grid.prod(dim=1) // 4).tolist()
    images = [torch.randn(rows, 8) for rows in lengths]
    full_features = torch.cat(images)
    native_prompt = assemble(native_pipeline, prompt, grid, full_features)
    native_offsets = [item.offsets[0] for item in native_prompt.mm_items]
    assert [end - start + 1 for start, end in native_offsets] == lengths
    assert native_prompt.mrope_positions.shape == (3, len(native_prompt.input_ids))
    for (start, end), (_, height, width) in zip(native_offsets, grids):
        image_positions = native_prompt.mrope_positions[:, start : end + 1]
        expected_grid = torch.cartesian_prod(
            torch.arange(1), torch.arange(height // 2), torch.arange(width // 2)
        ).T
        assert torch.equal(image_positions - image_positions[:, :1], expected_grid)
    scores_and_keys = [
        (torch.randperm(rows).float(), torch.randn(rows, 8)) for rows in lengths
    ]

    for ratio in (100, 70, 50):
        case = f"{image_count} images C{ratio}"
        encoded = [
            module.compress_image(image, scores, keys, ratio)
            for image, (scores, keys) in zip(images, scores_and_keys)
        ]
        selected_features, selections = zip(*encoded)
        compact_features = torch.cat(selected_features)
        compact = assemble(
            compressed_pipeline, prompt, grid, compact_features,
            visionzip_indices=selections,
        )
        kept_columns, expected_offsets = [], []
        cursor = 0
        for (start, end), selection in zip(native_offsets, selections):
            kept_columns.extend(range(cursor, start))
            offset_start = len(kept_columns)
            kept_columns.extend(start + index for index in selection)
            expected_offsets.append((offset_start, len(kept_columns) - 1))
            cursor = end + 1
        kept_columns.extend(range(cursor, len(native_prompt.input_ids)))
        compact_offsets = [item.offsets[0] for item in compact.mm_items]
        assert compact.input_ids == [native_prompt.input_ids[i] for i in kept_columns]
        assert compact_offsets == expected_offsets
        torch.testing.assert_close(
            compact.mrope_positions, native_prompt.mrope_positions[:, kept_columns],
            rtol=0, atol=0,
            msg=lambda detail: f"{case} M-RoPE coordinates: {detail}",
        )
        removed = len(native_prompt.input_ids) - len(compact.input_ids)
        torch.testing.assert_close(
            compact.mrope_position_delta, native_prompt.mrope_position_delta + removed,
            rtol=0, atol=0,
            msg=lambda detail: f"{case} M-RoPE delta: {detail}",
        )
        for item, feature in zip(compact.mm_items, selected_features):
            torch.testing.assert_close(item.precomputed_embeddings, feature, rtol=0, atol=0)

        if ratio == 100:
            assert removed == 0 and compact_offsets == native_offsets
            assert compact.input_ids == native_prompt.input_ids
        else:
            assert removed > 0
        for step in (1, 2, 5):
            absolute = int(native_prompt.mrope_positions.max()) + step
            expected = torch.full((3, 1), absolute, dtype=native_prompt.mrope_positions.dtype)
            for pipeline, output in (
                (native_pipeline, native_prompt), (compressed_pipeline, compact)
            ):
                actual = pipeline["_expand_mrope_from_input"](
                    None, output, len(output.input_ids) + step
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        if image_count == 1 and ratio == 50:
            for wrong_features in (compact_features[:-1], torch.cat(
                (compact_features, compact_features[:1])
            )):
                try:
                    assemble(
                        compressed_pipeline, prompt, grid, wrong_features,
                        visionzip_indices=selections,
                    )
                except ValueError as error:
                    assert "received rows do not match" in str(error)
                else:
                    raise AssertionError("incorrect feature count reached the language model")

native_source = "separate B source" if native_root != root else "native branch in C source"
print(f"PASS: real Qwen2.5-VL M-RoPE against {native_source}, "
      "1/4/8 images through C100/C70/C50, including grid 1x78x138.")

native_features = torch.randn(10, 32)
staged.clear()
ctx.aux_data.clear()
native = ns["_stage_embeddings"](
    fake_encoder, ctx, requests, native_features, keep_on_gpu=False
)
assert [r[1] for r in native] == [4, 6]
assert all(not hasattr(p, "visionzip_indices") for p in staged)
native_multi = MultiModalEmbeddingData.from_embedding_data(staged[0])
assert "visionzip_indices" not in native_multi.get_mm_extra_meta()
torch.testing.assert_close(
    native_multi.get_embedding(is_concat=True)[Modality.IMAGE], native_features[:4]
)
print(
    "PASS: attention mass parity, 50/70/100, malformed metadata, batched "
    "staging, out-of-order parts, received row validation, "
    "prepublished row counts and dtype, and native B staging."
)
