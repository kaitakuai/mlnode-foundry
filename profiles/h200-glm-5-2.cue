// Profile: H200 Hopper (×8) + GLM-5.2 FP8 — vllm-poc PLUGIN base.
//
// Composition: #BaseProfile & H200 & GLM_5_2 with TP=8, on the vllm-poc PLUGIN base
// (residual vLLM 0.23 + gonka-poc; ADR-0013). Source of truth:
// experiments/2026-06/glm-5.2-fp8-8xh200.
//
// HOPPER (sm_90) IS DIFFERENT from Blackwell (b200/b300, sm_100) — three
// load-bearing changes vs the b200/b300 GLM profiles:
//   1. kv-cache-dtype = fp8. On sm_90 GLM-DSA the attention backend is
//      FLASHMLA_SPARSE, where 'fp8' is the supported alias for fp8_ds_mla (a
//      656-byte compressed MLA KV format) — NOT the plain fp8_e4m3 that
//      FLASHMLA_SPARSE REJECTS ('No valid attention backend'). Pasha verified
//      --kv-cache-dtype fp8 + --max-num-batched-tokens 16384 on the built H200
//      image (2026-06-26); the compressed KV gives more headroom for the 400000
//      context. (The earlier experiment baked auto/bf16 for PoC; fp8 is the
//      post-test update. NONCE NOTE: fp8 KV changes the attention math vs bf16 — the
//      H200↔B200 L2-PASS was on bf16, so re-confirm nonce L2 on fp8 if mining.)
//   2. DeepGEMM OFF (triton MoE). DeepGEMM is a Blackwell sm_100 path; on Hopper
//      it is off, so there is no DeepGEMM linear crash and NO VLLM_DISABLED_KERNELS
//      / E8M0. The FP8 MoE auto-selector must be pinned to triton (env
//      VLLM_USE_FLASHINFER_MOE_FP8=0 + CLI --moe-backend triton) or it regresses to
//      the slower FLASHINFER_CUTLASS — same lesson as h200-minimax.
//   3. tool-call-parser glm45 (NOT glm47), --enable-expert-parallel ON. EP is unique
//      to the H200 experiment.
//
// Compilation NOT forced (compiled inference + eager PoC via gonka-poc
// skip_compiled; --enforce-eager is the operator pure-mining exception). NOTE the
// H200 L2-PASS vs B200 was collected on a FULL-EAGER engine; the shipped image mines
// on a compiled engine with skip_compiled PoC — re-confirm H200 nonce L2 on the
// compiled engine before trusting cross-hw validity. max-model-len 400000 is the
// operator directive; H200 bf16 KV holds ~553K at gmu0.90 so 400000 is comfortable.
package profiles

import (
	"list"
	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

h200_glm_5_2: #BaseProfile & bases.H200 & bases.GLM_5_2 & {
	identity: {
		axes: {
			gpu:            "h200"
			model:          "glm"
			model_revision: "5-2"
			// quant axis omitted — GLM-5.2 ships only as FP8 (zai-org/GLM-5.2-FP8).
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.23.0"
			rev:    1
		}
	}
	mode: "kaitakuai-base"
	// H200 base ships NO hw_patches (sm_90 needs none of the sm_100 driver/header
	// fixes); add ONLY cold-start-tolerance for the slow 753B+DSA init. Mirrors
	// h200-minimax-m2-7.cue's list.Concat shape (poc-householder-compile there is a
	// fat-fork fragment, NOT used on the plugin base).
	hw_patches: list.Concat([bases.H200.hw_patches, ["cold-start-tolerance"]])
	runner_patch: "h200-glm-5-2-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// DeepGEMM OFF on Hopper sm_90 → triton MoE. The Stage 2/3 base bakes
		// VLLM_USE_FLASHINFER_MOE_FP8=1, which on sm_90 routes MoE through the slower
		// FLASHINFER_CUTLASS; pin it to 0 here so the env layer agrees with triton out
		// of the box (the runner-patch also forces CLI --moe-backend triton). NO
		// VLLM_DISABLED_KERNELS / E8M0 — DeepGEMM is fully off, no linear kernel to
		// route to Cutlass.
		VLLM_USE_DEEP_GEMM:          "0"
		VLLM_MOE_USE_DEEP_GEMM:      "0"
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
		// VLLM_RUNNER_TIMEOUT / WATCHER_GRACE_FIRST_HEALTHY come from the GLM_5_2
		// base; VLLM_USE_V1 from the H200 base.
	}
	// Compiled inference (vLLM default cudagraph) + eager PoC via gonka-poc
	// skip_compiled — NOT forced (operator passes --enforce-eager for the exceptional
	// pure-mining eager case). Display from the GLM_5_2 base.
	runtime_defaults: {
		// From experiments/2026-06/glm-5.2-fp8-8xh200; forced via the runner-patch,
		// reproduced here for dashboard display.
		tensor_parallel_size:    8
		gpu_memory_utilization:  0.90
		max_model_len:           400000
		max_num_batched_tokens:  16384
		max_num_seqs:            128
		kv_cache_dtype:          "fp8"
		tool_call_parser:        "glm45"
		reasoning_parser:        "glm45"
		moe_backend:             "triton"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		enable_expert_parallel:  true
		// NO attention_backend: GLM-5.2 is DSA — FLASHMLA_SPARSE auto-selects on
		// sm_90; do NOT pin CUTLASS_MLA.
	}
	description: "H200 Hopper SXM5 (×8) + GLM-5.2 FP8 (TP=8, eager PoC / compiled inference, triton MoE, fp8_ds_mla KV) — vllm-poc PLUGIN base"
	notes: """
		GLM-5.2 FP8 on H200 (Hopper sm_90), from experiments/2026-06/glm-5.2-fp8-8xh200
		+ Pasha's post-test directive (2026-06-26: kv fp8 + mnbt 16384 on the built image).
		Config: TP=8, gpu-memory-utilization 0.90 (sparse_decode_fwd needs ~2 GB/card
		scratch; gmu 0.97 OOMs), max-model-len 400000 (operator-forced), kv-cache-dtype
		fp8 (= FLASHMLA_SPARSE fp8_ds_mla alias, compressed), max-num-batched-tokens
		16384, max-num-seqs 128, glm45 tool-call + glm45 reasoning parsers,
		--enable-expert-parallel.

		Hopper-specific (vs b200/b300 GLM):
		  - kv-cache-dtype=fp8: on sm_90 GLM-DSA the backend is FLASHMLA_SPARSE, where
		    'fp8' is the supported alias for fp8_ds_mla (656-byte compressed MLA KV) —
		    plain fp8_e4m3 is REJECTED ('No valid attention backend'). Pasha-verified on
		    the built image. NONCE: fp8 changes attention math vs the bf16 the H200↔B200
		    L2-PASS used — re-confirm nonce L2 on fp8 if mining.
		  - DeepGEMM OFF → triton MoE (env VLLM_USE_FLASHINFER_MOE_FP8=0 + CLI
		    --moe-backend triton); no VLLM_DISABLED_KERNELS / E8M0.
		  - tool-call-parser glm45 (NOT glm47), --enable-expert-parallel ON.
		  - attention backend FLASHMLA_SPARSE auto-selected (NOT pinned).

		Compilation/eager policy identical to b200/b300-glm (compiled inference + eager
		PoC via skip_compiled; --enforce-eager is the pure-mining exception).

		Throughput (experiment, eager): PoC 576 nonces/min (−43% vs B200's 1016;
		B200 higher bandwidth + sm_100 FLASHMLA). H200↔B200 FP8 nonces L2-PASS (mean
		0.188, 0.7% mismatch) — but that PASS was on a FULL-EAGER engine; the shipped
		image mines on a COMPILED engine (skip_compiled PoC). RE-CONFIRM H200 nonce L2
		on the compiled engine + re-benchmark on the foundry-built leaf before
		production. The earlier H200 experiment ran on the kimi image with overrides;
		this is the dedicated H200 GLM image.
		"""
	tuning_notes: [
		{
			knob:     "kv_cache_dtype=fp8 (FLASHMLA_SPARSE fp8_ds_mla alias — NOT fp8_e4m3)"
			source:   "Pasha post-test directive (2026-06-26) on the built H200 image"
			reason:   "LOAD-BEARING: on sm_90 GLM-DSA the backend is FLASHMLA_SPARSE, where 'fp8' is the supported alias for fp8_ds_mla (656-byte compressed MLA KV); plain fp8_e4m3 is REJECTED ('No valid attention backend'). Pasha verified fp8 + mnbt 16384 on hardware. Compressed KV → more headroom for 400000. NONCE: fp8 changes attention math vs the bf16 the H200↔B200 L2-PASS used — re-confirm nonce L2 on fp8 if mining."
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "MoE backend=triton (DeepGEMM OFF on sm_90)"
			source:   "glm-5.2-fp8-8xh200 config.env + h200-minimax precedent"
			reason:   "DeepGEMM is a Blackwell path; on Hopper it is off. The Stage 2/3 base bakes VLLM_USE_FLASHINFER_MOE_FP8=1 which routes sm_90 MoE through the slower FLASHINFER_CUTLASS — pin env to 0 AND force CLI --moe-backend triton so the selector cannot regress. No VLLM_DISABLED_KERNELS / E8M0 (nothing to route to Cutlass)."
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "tool_call_parser=glm45 + enable_expert_parallel"
			source:   "glm-5.2-fp8-8xh200 README (effective-flags + changed-vs-Kimi)"
			reason:   "H200 uses glm45 tool-call parser (B200/B300 use glm47) and --enable-expert-parallel (omitted on B200/B300). Top transcription hazard — verified verbatim against the experiment config block."
			added_at: "2026-06-26"
		},
		{
			knob:     "max_num_batched_tokens=16384"
			source:   "Pasha post-test directive (2026-06-26)"
			reason:   "Pasha-verified on the built H200 image alongside kv-cache-dtype=fp8. (The pre-test experiment used 65536 to dodge a 131072+bf16 KV-profiling OOM; with the compressed fp8_ds_mla KV the footprint is smaller, and 16384 also matches B200/B300.)"
			added_at: "2026-06-26"
		},
		{
			knob:     "max_model_len=400000 (operator-forced)"
			source:   "Pasha directive + glm-5.2-fp8-8xh200"
			reason:   "400000 on all GLM GPUs. H200 SAFE: experiment measured ~553,668-token bf16 KV cache at gmu0.90; 400000 is below the proven 450000 ceiling, and the compressed fp8_ds_mla KV (now baked) gives even more headroom."
			added_at: "2026-06-26"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-fp8-8xh200"
			reason:   "PoC 576 nonces/min eager (−43% vs B200). H200↔B200 FP8 L2-PASS (0.7% mismatch) but collected on a FULL-EAGER engine — RE-CONFIRM L2 on the compiled engine (skip_compiled PoC) the shipped image uses, + re-benchmark on the foundry-built leaf before production."
			severity: "warning"
			added_at: "2026-06-26"
		},
	]
}
