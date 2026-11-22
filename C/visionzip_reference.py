"""Run the single-image VisionZip author reference on a frozen quality trace."""

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from io import BytesIO
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1_605_632


class UnsupportedAspectRatio(ValueError):
    """The frozen Qwen preprocessing rejects this image's aspect ratio."""


@contextmanager
def load_model_class(ratio, author_source):
    if ratio == 100:
        from transformers import Qwen2_5_VLForConditionalGeneration

        yield Qwen2_5_VLForConditionalGeneration
        return

    source = author_source.read_text(encoding="utf-8")
    selected = "dominant_num = int(0.65 * attn_logits.size(0))"
    if source.count(selected) != 1 or "contextual_num = max(int(0.05 * attn_logits.size(0)),1)" not in source:
        raise ValueError("The frozen VisionZip source does not have the expected ratios")
    if ratio == 50:
        source = source.replace(selected, selected.replace("0.65", "0.45"))
    with TemporaryDirectory(prefix="visionzip-author-") as directory:
        staged = Path(directory) / author_source.name
        staged.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("qwen2_5vl_visionzip", staged)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module.Qwen2_5_VLForConditionalGeneration


def decode_image(row):
    from PIL import Image

    if row.get("task") not in ("ocrbench", "realworldqa") or len(row["images"]) != 1:
        raise ValueError("The author reference accepts single-image OCRBench or RealWorldQA rows")
    if row["ignore_eos"]:
        raise ValueError("The author reference expects quality traces with natural EOS")
    header, data = row["images"][0].split(",", 1)
    if header != "data:image/png;base64":
        raise ValueError("Expected frozen PNG image bytes")
    raw = base64.b64decode(data, validate=True)
    if row["image_digests"] != [hashlib.sha256(raw).hexdigest()]:
        raise ValueError(f"Image hash mismatch for {row['id']}")
    with Image.open(BytesIO(raw)) as image:
        return image.convert("RGB")


def encode_inputs(processor, row, image):
    from qwen_vl_utils import process_vision_info
    from qwen_vl_utils.vision_process import MAX_RATIO

    width, height = image.size
    if max(width, height) > MAX_RATIO * min(width, height):
        raise UnsupportedAspectRatio(f"{width}×{height} exceeds Qwen aspect-ratio limit {MAX_RATIO}")
    content = [
        {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": row["text"]},
    ]
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info([messages])
    return processor(text=[prompt], images=images, padding=True, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=int, choices=(100, 70, 50), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--author-source",
        type=Path,
        help="Author source path from the pinned checkout (required for 70 or 50)",
    )
    parser.add_argument("--attention", choices=("eager", "flash_attention_2"), default="eager")
    args = parser.parse_args()
    if args.ratio < 100 and args.author_source is None:
        parser.error("--author-source is required for --ratio 70 or 50")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from transformers import AutoProcessor

    with args.trace.open(encoding="utf-8") as trace:
        first = json.loads(trace.readline())
        image = decode_image(first)
        processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("The author reference expects exactly one visible GPU")
        with load_model_class(args.ratio, args.author_source) as model_class, \
                args.output.open("x", encoding="utf-8") as output, torch.inference_mode():
            model = model_class.from_pretrained(
                args.model,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attention,
                device_map={"": 0},
            ).eval()
            for index, row in enumerate(chain([first], map(json.loads, trace)), 1):
                answer = {
                    "request_id": row["id"],
                    "dataset_revision": row["dataset_revision"],
                    "doc_id": row["doc_id"],
                    "image_digests": row["image_digests"],
                    "group": f"author{args.ratio}",
                }
                try:
                    if index > 1:
                        image = decode_image(row)
                    inputs = encode_inputs(processor, row, image).to("cuda")
                    started = perf_counter()
                    sequences = model.generate(
                        **inputs, max_new_tokens=row["output_tokens"], do_sample=False
                    )
                    generated = sequences[:, inputs.input_ids.shape[1] :]
                    text = processor.batch_decode(
                        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    answer.update(
                        status="complete",
                        content_chunks=[[perf_counter() - started, text]],
                        output_tokens=int(generated.shape[1]),
                    )
                except UnsupportedAspectRatio as error:
                    answer.update(status="rejected", error=str(error))
                except Exception as error:
                    answer.update(status="error", error=str(error))
                    output.write(json.dumps(answer, ensure_ascii=False) + "\n")
                    raise
                output.write(json.dumps(answer, ensure_ascii=False) + "\n")
                output.flush()
                if index == 1 or index % 50 == 0:
                    print(f"author{args.ratio}: handled {index} rows", flush=True)


if __name__ == "__main__":
    main()
