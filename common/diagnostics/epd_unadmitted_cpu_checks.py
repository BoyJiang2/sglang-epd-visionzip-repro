"""Regression for the E/PD request retired before it receives its token layout."""

import ast
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace


class ReqStub:
    def __init__(self, rid, text, ids, sampling_params, **kwargs):
        self.origin_input_ids = ids
        self.origin_input_ids_unpadded = ids
        self.sampling_params = sampling_params
        self.http_worker_ipc = kwargs["http_worker_ipc"]
        self.output_ids = array("q")


def main(repo):
    srt = Path(repo) / "python/sglang/srt"
    receiver = ast.parse(
        (srt / "disaggregation/encoder/receiver.py").read_text(encoding="utf-8")
    )
    detokenizer = ast.parse(
        (srt / "managers/incremental_detokenize.py").read_text(encoding="utf-8")
    )
    owner = next(node for node in receiver.body if getattr(node, "name", None) == "MMReceiverBase")
    create_req = next(node for node in owner.body if getattr(node, "name", None) == "create_req")
    # Use the relevant Req constructor state without importing the GPU runtime.
    assert isinstance(create_req.body[0], ast.ImportFrom)
    create_req.body.pop(0)
    header = next(
        node for node in detokenizer.body if getattr(node, "name", None) == "build_detokenize_header"
    )
    nodes = ast.parse("from __future__ import annotations").body + [create_req, header]
    ns = dict(Req=ReqStub, array=array, PROMPT_TAIL_LEN=5, DetokenizeHeader=SimpleNamespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), __file__, "exec"), ns)

    fields = (
        "rid", "input_text", "return_logprob", "top_logprobs_num", "token_ids_logprob",
        "stream", "lora_id", "input_embeds", "custom_logit_processor", "require_reasoning",
        "return_hidden_states", "return_routed_experts", "routed_experts_start_len",
        "bootstrap_host", "bootstrap_port", "bootstrap_room", "routed_dp_rank",
        "disagg_prefill_dp_rank", "priority", "extra_key", "cache_salt", "http_worker_ipc",
    )
    params = SimpleNamespace(
        skip_special_tokens=True, spaces_between_special_tokens=True, no_stop_trim=False
    )
    recv = SimpleNamespace(**dict.fromkeys(fields), sampling_params=params, input_ids=None)
    mocked_receiver = SimpleNamespace(
        scheduler=SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id=None, token_id_vocab_size=30),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            disaggregation_mode="null",
            dllm_config=None,
            tokenizer=None,
        )
    )
    retired = ns["create_req"](mocked_receiver, recv)
    assert ns["build_detokenize_header"](retired).prompt_tail_ids == array("q")
    assert retired.origin_input_ids == array("q")

    recv.input_ids = array("q", [2, 3, 4])
    known = ns["create_req"](mocked_receiver, recv)
    assert known.origin_input_ids is recv.input_ids
    assert ns["build_detokenize_header"](known).prompt_tail_ids == recv.input_ids
    print("PASS: a missing token layout can retire; existing input IDs stay unchanged")


if __name__ == "__main__":
    main(sys.argv[1])
