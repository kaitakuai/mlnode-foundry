// Profile: B200 Blackwell (×4) + Kimi-K2.6 INT4 — release-matrix leaf.
//
// Composition: #OverlayProfile & B200 & KIMI_INT4 with TP=4, overlaid on cortima's
// published release mlnode (vLLM 0.25.1 residual + gonka-poc plugin; ADR-0013).
// Migrated in place from
// the fat-fork 0.20 rev=2 tune (which this validated config came from). Config is
// the B200 Kimi experiment kimi_k26_int4_4xb200_q-int4-k2: TP=4, CUDA graphs
// (compilation mode=3 / FULL_AND_PIECEWISE), gmu 0.93, batch=32 (mnbt 32768),
// max-model-len 120000, CUTLASS_MLA, expert-parallel, kimi_k2 parsers,
// mm-encoder-tp-mode data — ~2240 nonces/min @ batch=32 on a single 4×B200.
//
// COMPILATION vs b300-kimi: b300-kimi FORCES EAGER (cudagraph hangs at batch=128 on
// Kimi MLA on B300). B200 runs the SMALLER batch=32 envelope (178/183 GiB/GPU), and
// at batch=32 CUDA graphs work fine (the experiment ran them). The experiment showed
// CUDA graphs do NOT change PoC throughput (PoC is a separate forward path), so we
// keep mode=3 FULL_AND_PIECEWISE for faster inference at no PoC cost — UNLIKE b300.
// The PoC forward still runs eager on its own via gonka-poc skip_compiled.
//
// Plugin differences vs the fat-fork b200-kimi rev=2:
//   - env adds MLNODE_VLLM_MODULE → the runner launches the composed gonka-poc
//     entrypoint; runner_patch forces --worker-extension-cls (worker-side PoC).
//   - HF_HUB_OFFLINE=1 baked (Kimi tokenizer workaround; pre-stage weights).
//   - --enforce-eager removed (conflicts with the forced --compilation-config).
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b200_kimi_k2_6: #OverlayProfile & bases.B200 & bases.KIMI_INT4 & bases.KIMI_INT4_FLASHINFER_MOE & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "kimi"
			model_revision: "k2-6"
			// quant axis omitted — Kimi-K2.6 ships only as INT4 (compressed-tensors
			// W4A16). Mirrors b300-kimi-k2-6.cue.
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			upstream: "3.0.14-post2-vllm0.25.1-rc3"
			rev:      1
		}
	}
	mode: "upstream-overlay"
	base: {
		// Cortima's PUBLISHED release image — see b300-kimi-k2-6.cue on the switch.
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:450983bbef31c8e19b8d24edb00c17520af7cc4fb0d186943f3ac3dec4dad387"
		upstream_version: "3.0.14-post2-vllm0.25.1-rc3"
	}
	// The fat-fork's poc-householder-compile is NOT referenced — it edited the
	// monolith's vllm/poc/ tree, absent on the plugin base.
	hw_patches: list.Concat([bases.B200.hw_patches, bases.GONKA_MLNODE_PATCHES])
	runner_patch: "b200-kimi-k2-6-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint
		// (build_app + PoC router + gating) instead of stock api_server (ADR-0013).
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// Kimi-K2.6's newer revision ships a buggy tokenization_kimi_fast.py;
		// HF_HUB_OFFLINE forces the cached (pre-staged) tokenizer. Requires weights
		// pre-staged in the HF cache (true on fleet nodes; pre-stage before a
		// standalone run). See [[project_kimi_hf_hub_offline]].
		HF_HUB_OFFLINE:      "1"
		// Long Kimi-K2.6 INT4 load (~530 GiB on TP=4) + MLA warmup.
		VLLM_RUNNER_TIMEOUT: "3600"
	}
	runtime_defaults: {
		// CUDA graphs (override the KIMI_INT4 base eager default: mode=0/NONE →
		// mode=3/FULL_AND_PIECEWISE). Works at B200's batch=32 (unlike B300 batch=128
		// where cudagraph hangs → b300-kimi forces eager). Forced via the runner-patch;
		// reproduced here for the dashboard. PoC forward is eager via gonka-poc
		// skip_compiled regardless.
		compilation_mode:        3
		cudagraph_mode:          "FULL_AND_PIECEWISE"
		tensor_parallel_size:    4
		gpu_memory_utilization:  0.93
		max_num_batched_tokens:  32768
		max_model_len:           120000
		max_num_seqs:            32
		attention_backend:       "CUTLASS_MLA"
		tool_call_parser:        "kimi_k2"
		reasoning_parser:        "kimi_k2"
		mm_encoder_tp_mode:      "data"
		logprobs_mode:           "processed_logprobs"
		trust_remote_code:       true
		enable_auto_tool_choice: true
		enable_expert_parallel:  true
	}
	description: "B200 Blackwell SXM (×4) + Kimi-K2.6 INT4 (TP=4, CUDA graphs, EP, batch=32) — vllm-poc PLUGIN base (gonka-poc entrypoint + worker extension)"
	notes: """
		B200 Kimi-K2.6 INT4 on the 0.25.1 release base. Config from the validated B200
		Kimi experiment (kimi_k26_int4_4xb200_q-int4-k2): TP=4, CUDA graphs
		(compilation mode=3 / FULL_AND_PIECEWISE), gpu-memory-utilization 0.93,
		batch=32 (max-num-batched-tokens 32768 = 32×1024), max-num-seqs 32,
		max-model-len 120000, CUTLASS_MLA, expert-parallel, kimi_k2 tool/reasoning
		parsers, mm-encoder-tp-mode data, INT4 FlashInfer MoE (KIMI_INT4 base).

		Memory envelope (vs b300-kimi's batch=128/uncapped 256K): B200's 178-183
		GiB/GPU caps the cudagraph capture, so batch=32 + max-model-len 120000
		(prompts >120k truncate). Model loads ~141 GiB/card; ~14 GiB KV pool at
		gmu 0.93. ~2240 nonces/min @ batch=32 on a single 4×B200 instance.

		Plugin flip vs the fat-fork rev=2:
		  - MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router (composed server).
		  - runner-patch forces --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
		    (worker-side PoC via collective_rpc), plus the experiment tune above.
		  - --enforce-eager removed (conflicts with --compilation-config; mode=3 is
		    the compiled path, PoC-forward eager comes from gonka-poc skip_compiled).
		  - HF_HUB_OFFLINE=1 baked (Kimi tokenizer workaround) — pre-stage weights.

		COMPILATION differs from b300-kimi (which forces EAGER because cudagraph hangs
		at batch=128 on Kimi MLA on B300). At B200's batch=32 cudagraph works, and the
		experiment showed CUDA graphs do not change PoC throughput (separate forward
		path) — so mode=3 keeps faster inference at no PoC cost.

		NOT yet re-validated on the 0.25.1 release base — re-benchmark on 4×B200
		(start clean, no MLA assert, PoC ~2240/min @ batch=32, INT4 nonces L2-valid
		under the Kimi gate) before treating as production.
		"""
	tuning_notes: [
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/kimi_k26_int4_4xb200_q-int4-k2/README.md"
			reason:   "Hardware validation on 4×B200 (Vast.ai): ~2240 nonces/min @ batch=32, no MLA assert (after kaitakuai/vllm#9 seq_lens_cpu_upper_bound restore), INT4 nonces L2-PASS vs B300 (0.1888, 1.5%) and B200-k5 (0.1874, 2.5%). Config is fat-fork-proven; NOT yet re-benchmarked on the 0.25.1 release base — re-confirm on 4×B200 before production."
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "compilation_mode=3"
			source:   "kimi_k26_int4_4xb200_q-int4-k2 experiment"
			reason:   "CUDA graphs (mode 3 + FULL_AND_PIECEWISE). Unlike b300-kimi (forced eager — cudagraph hangs at batch=128 on B300), B200 runs batch=32 where cudagraph works; the experiment confirmed CUDA graphs do not change PoC throughput (PoC is a separate forward path), so mode=3 keeps faster inference at no PoC cost. PoC forward eager via gonka-poc skip_compiled."
			added_at: "2026-06-26"
		},
		{
			knob:     "max_num_batched_tokens=32768"
			source:   "kimi_k26_int4_4xb200_q-int4-k2 experiment"
			reason:   "Caps PoC effective batch at 32 (32×1024). Batch 64/128 exceed the token budget and OOM at B200's 178 GiB/GPU under cudagraph FULL_AND_PIECEWISE. Above upstream chunked-prefill default 8192 — perf opt, no baseline violation."
			added_at: "2026-06-26"
		},
		{
			knob:     "max_model_len=120000"
			source:   "kimi_k26_int4_4xb200_q-int4-k2 experiment"
			reason:   "Capped below Kimi-K2.6's native 262144 (~54% cut). Cudagraph FULL captures KV shapes at 256k that OOM at B200's 178 GiB/GPU. Prompts >120k truncate. (B300 leaves 256K uncapped — 275 GiB/GPU has room.)"
			severity: "warning"
			added_at: "2026-06-26"
		},
		{
			knob:     "enable_expert_parallel=True"
			source:   "kimi_k26_int4_4xb200_q-int4-k2 experiment"
			reason:   "EP=4 over the TP=4 sharding (legacy B300 Kimi showed +5% EP gain). Enabled in the validated B200 config."
			added_at: "2026-06-26"
		},
	]
}
