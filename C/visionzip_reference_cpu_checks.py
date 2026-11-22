"""Check the frozen VisionZip eager decoder with tiny weights and synthetic vision features."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen2_5_VLConfig

import visionzip_reference as reference


class DummyVision(nn.Module):
    dtype = torch.float32

    def forward(self, pixel_values, grid_thw):
        tokens = int(grid_thw.prod()) // 4
        random = torch.Generator().manual_seed(tokens)
        embeddings = torch.randn(tokens, 32, generator=random)
        keys = torch.randn(1, tokens, 8, generator=random)
        return embeddings, torch.linspace(0, 1, tokens), keys


def check_generation(author_source, ratio, shapes, verify_layout=True):
    config = Qwen2_5_VLConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, sliding_window=None,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
        vision_config={"depth": 1, "hidden_size": 32, "out_hidden_size": 32,
                       "intermediate_size": 64, "num_heads": 2, "fullatt_block_indexes": [0]},
        pad_token_id=0, bos_token_id=1, eos_token_id=127,
        vision_start_token_id=120, vision_end_token_id=122, image_token_id=121, video_token_id=123,
    )
    config._attn_implementation = "eager"
    with reference.load_model_class(ratio, author_source) as model_class:
        model = model_class(config).eval()
        model.visual = DummyVision()
        for height, width in shapes:
            image_tokens = height * width // 4
            input_ids = torch.tensor([[1, 120] + [121] * image_tokens + [122] + [8] * 25])
            grid = torch.tensor([[1, height, width]])
            mask = torch.ones_like(input_ids)
            original_length = input_ids.shape[1]
            dominant = int((0.65 if ratio == 70 else 0.45) * image_tokens)
            contextual = max(int(0.05 * image_tokens), 1)
            compact_length = original_length - image_tokens + dominant + contextual
            lengths = []

            def check_decoder(_module, _args, kwargs):
                step = len(lengths)
                positions = kwargs["cache_position"]
                past_length = kwargs["past_key_values"].get_seq_length()
                rope = kwargs["position_ids"]
                attention = kwargs["attention_mask"]
                if step == 0:
                    original_rope, _ = model.get_rope_index(input_ids, grid, attention_mask=mask)
                    assert past_length == 0 and kwargs["inputs_embeds"].shape[1] == compact_length
                    assert torch.equal(positions, torch.arange(compact_length))
                    assert torch.equal(rope, original_rope[..., model._visionzip_prompt_keep])
                else:
                    assert past_length == compact_length + step - 1 and positions.tolist() == [past_length]
                    assert torch.all(rope == original_length + step - 1 + model.rope_deltas.item())
                assert attention.shape == (1, past_length + len(positions))
                lengths.append(attention.shape[-1])

            if verify_layout:
                hook = model.model.register_forward_pre_hook(check_decoder, with_kwargs=True)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids, attention_mask=mask, pixel_values=torch.ones(1),
                    image_grid_thw=grid, do_sample=False, min_new_tokens=4, max_new_tokens=4,
                )
            if verify_layout:
                hook.remove()
                assert lengths == list(range(compact_length, compact_length + 4))
                assert output.shape == (1, original_length + 4)
                print(f"{ratio}% {original_length} → {compact_length}, cache lengths {lengths}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("author_source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    with patch.object(reference, "compact_author_cache", side_effect=lambda source: source):
        try:
            check_generation(args.author_source, 70, [(32, 36)], verify_layout=False)
        except RuntimeError as error:
            assert "229" in str(error) and "316" in str(error), str(error)
        else:
            raise AssertionError("Expected a 229 versus 316 shape error from the original eager path")
    for ratio in (70, 50):
        check_generation(args.author_source, ratio, [(32, 36), (24, 32)])


if __name__ == "__main__":
    main()
