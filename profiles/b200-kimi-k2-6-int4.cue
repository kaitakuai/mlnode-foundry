// Profile: B200 Blackwell + Kimi-K2.6 INT4 (experimental tune k=2)
//
// Most-promising tuning candidate as of 2026-05-19 — moves off rev=1's eager
// defaults onto a compiled cudagraph path with a smaller batch envelope.
// Args are baked into mlnode runner.py via tools/runner-patches/b200-kimi-k2-6-int4.py
// so the image is reproducible end-to-end for hardware validation (operator
// just `docker run` and the chain-broadcast args get overridden by the patcher).
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b200_kimi_k2_6_int4: #BaseProfile & bases.B200 & bases.KIMI_INT4 & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "kimi"
			model_revision: "k2-6"
			quant:          "int4"
		}
		version: {
			mlnode: "0.2.13"
			vllm:   "0.20.0"
			// rev=2: experimental tune with compilation mode=3 + FULL_AND_PIECEWISE
			// cudagraph + expert-parallel + smaller batch (32 vs 128). rev=1 was
			// the eager-only baseline ported from legacy b200-k5-kimi-1.
			rev: 2
		}
	}
	mode: "kaitakuai-base"
	// Same hw-patches as rev=1; experimental tune doesn't add or remove any.
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
	]
	runner_patch: "b200-kimi-k2-6-int4"
	env: {
		VLLM_RUNNER_TIMEOUT: "3600"
	}
	runtime_defaults: {
		// Override KIMI_INT4 base defaults (mode=0 / NONE → mode=3 / FULL_AND_PIECEWISE).
		compilation_mode:       3
		cudagraph_mode:         "FULL_AND_PIECEWISE"
		// Forced via runner-patch — values reproduced here for dashboard display.
		tensor_parallel_size:   4
		gpu_memory_utilization: 0.93
		max_num_batched_tokens: 32768
		max_model_len:          120000
		max_num_seqs:           32
		attention_backend:      "CUTLASS_MLA"
		tool_call_parser:       "kimi_k2"
		reasoning_parser:       "kimi_k2"
		mm_encoder_tp_mode:     "data"
		logprobs_mode:          "processed_logprobs"
		trust_remote_code:      true
		enable_auto_tool_choice: true
		enable_expert_parallel:  true
	}
	description: "B200 Blackwell SXM (×4) + Kimi-K2.6 INT4 — experimental tune k=2 (compiled + EP + smaller batch)"
	notes: """
		Replaces rev=1's eager defaults with:
		  - compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
		  - --enable-expert-parallel
		  - --gpu-memory-utilization 0.93 (was 0.95 / 0.85 baseline)
		  - --max-num-batched-tokens 32768 (was 131072)
		  - --max-num-seqs 32 (was 128)
		  - --max-model-len 120000 (caps below native 256k to fit cudagraph capture)
		  - removed --enforce-eager
		Operator handoff: docker pull + run, the runner-patch forces these args
		over anything chain epoch_models broadcasts.
		"""
	tuning_notes: [
		{
			knob:     "compilation-config.mode=3,cudagraph_mode=FULL_AND_PIECEWISE"
			source:   "operator iteration 2026-05-19 (handoff to colleague for benchmark)"
			reason:   "Replaces rev=1 mode=0/NONE eager defaults with compiled+cudagraph path. Throughput claim untested; this image is the test."
			severity: "warning"
			added_at: "2026-05-19"
		},
		{
			knob:     "enable-expert-parallel"
			source:   "operator iteration 2026-05-19"
			reason:   "Adds EP=4 over the TP=4 sharding. Legacy B300 Kimi config showed +5% EP gain; replicate on B200 here."
			added_at: "2026-05-19"
		},
		{
			knob:     "max-num-batched-tokens=32768,max-num-seqs=32,max-model-len=120000"
			source:   "operator iteration 2026-05-19"
			reason:   "Smaller batch envelope is required when cudagraph captures large shapes; 131072 baseline OOMs cudagraph FULL on Kimi MLA on B200."
			severity: "warning"
			added_at: "2026-05-19"
		},
	]
}
