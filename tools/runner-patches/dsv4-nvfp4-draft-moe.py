"""Keep DSpark draft experts off the NVFP4 path — S4 form of kaitakuai/vllm#20.

NVFP4 conversions of DeepSeek-V4 convert only the target model's experts and
leave the DSpark draft (``mtp.*``) experts in MXFP4; ``Fp8Config`` never reads
the checkpoint's ``ignore`` list, so ``moe_quant_algo`` was applied to the
draft too and it emitted garbage — acceptance collapsed to ~1.2 tok/chunk.
Consult ``quantized_layers`` (the checkpoint's own per-layer map) instead.
Measured on 1xB300: 2.7-5.6x decode throughput, PoC unaffected, greedy output
byte-identical (see the PR for the full table).

That PR lands in our residual tree, which release-line images no longer build
from — they overlay gonka's published mlnode. Same edit, applied at Stage 4;
delete once the fix reaches gonka-ai/vllm release/v0.25.1 and a base carries
it. Referenced only by the b300 DeepSeek leaf (Pasha, 2026-08-10): NVFP4 is
the technically-primary variant on B300 alone.
"""

import importlib.util
import sys
from pathlib import Path

IMPORT_ANCHOR = "from typing import TYPE_CHECKING\n"
IMPORT_LINE = "import re\n"

INIT_ANCHOR = "        self._resolved_moe_quant_algo: str | None = None\n"
INIT_LINE = "        self._nvfp4_expert_prefixes: set[str] | None = None\n"

METHODS_ANCHOR = "    def _get_nvfp4_config(self)"
METHODS = '''    def _nvfp4_expert_prefix_set(self) -> set[str]:
        """Expert prefixes the checkpoint actually converted to NVFP4.

        NVFP4 conversions of V4 (both ``nvidia/DeepSeek-V4-Flash-NVFP4`` and the
        community 0731 port) convert only ``layers.0..N-1.ffn.experts`` and leave
        the DSpark draft (``mtp.*``) experts in the source MXFP4 representation.
        They declare that via ``ignore``, which ``Fp8Config`` does not read, so
        ``moe_quant_algo`` alone would apply NVFP4 to the draft experts too.
        ``quantized_layers`` is the authoritative per-layer map; empty means the
        checkpoint tells us nothing and the caller keeps the old behaviour.
        """
        if self._nvfp4_expert_prefixes is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
                quant_cfg = getattr(hf_config, "quantization_config", None) or {}
            except Exception:
                return set()
            layers = quant_cfg.get("quantized_layers") or {}
            self._nvfp4_expert_prefixes = {
                name
                for name, info in layers.items()
                if str((info or {}).get("quant_algo", "")).upper() == "NVFP4"
            }
        return self._nvfp4_expert_prefixes

    def _is_nvfp4_expert_layer(self, prefix: str) -> bool | None:
        """True/False per the checkpoint map, or None when it has no map."""
        converted = self._nvfp4_expert_prefix_set()
        if not converted:
            return None
        # Runtime prefixes are "model.layers.<i>.ffn.experts"; the map keys drop
        # the "model." Draft layers are built at index >= num_hidden_layers, but
        # a checkpoint may name them in mtp space instead - accept either.
        candidates = {prefix, prefix.removeprefix("model.")}
        match = re.search(r"layers\\.(\\d+)\\.(.*)$", prefix)
        if match is not None:
            try:
                n_layers = int(
                    get_current_vllm_config().model_config.hf_config.num_hidden_layers
                )
            except Exception:
                n_layers = None
            index = int(match.group(1))
            if n_layers is not None and index >= n_layers:
                candidates.add(f"mtp.{index - n_layers}.{match.group(2)}")
        return bool(candidates & converted)

'''

COND_OLD = """            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
"""
COND_NEW = """            if self.expert_dtype == "fp4":
                # Only take the NVFP4 path for experts the checkpoint really
                # converted. None => no per-layer map, keep the old behaviour.
                is_nvfp4 = self._is_nvfp4_expert_layer(prefix or "")
                if self.moe_quant_algo == "NVFP4" and is_nvfp4 is not False:
"""


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.stderr.write("ERROR: dsv4-nvfp4-draft-moe: vllm is not installed\n")
        return 1
    path = Path(spec.submodule_search_locations[0]) / "models/deepseek_v4/quant_config.py"
    src = path.read_text()

    if "_is_nvfp4_expert_layer" in src:
        print("dsv4-nvfp4-draft-moe: already patched; no-op")
        return 0
    for anchor, name in (
        (IMPORT_ANCHOR, "import"),
        (INIT_ANCHOR, "init"),
        (METHODS_ANCHOR, "methods"),
        (COND_OLD, "condition"),
    ):
        if src.count(anchor) != 1:
            sys.stderr.write(
                f"ERROR: dsv4-nvfp4-draft-moe: {name} anchor not found exactly once "
                f"in {path}. The quant config may have been refactored — re-verify "
                "against kaitakuai/vllm#20.\n"
            )
            return 1

    src = src.replace(IMPORT_ANCHOR, IMPORT_LINE + IMPORT_ANCHOR, 1)
    src = src.replace(INIT_ANCHOR, INIT_ANCHOR + INIT_LINE, 1)
    src = src.replace(METHODS_ANCHOR, METHODS + METHODS_ANCHOR, 1)
    src = src.replace(COND_OLD, COND_NEW, 1)

    path.write_text(src)
    print(f"dsv4-nvfp4-draft-moe: patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
