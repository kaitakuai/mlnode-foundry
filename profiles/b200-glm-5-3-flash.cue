package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

// GLM-5.3-Flash on Cortima's published mlnode 3.1.0 (gonka-ai/vllm
// release/v0.28.0-glm53 after #104-#106 and #108, gonka-poc v0.1.6). Its S2
// shares the first 32 layers with our validated
// kaitakuai/vllm-poc:glm53-poc-v4-ed8873884 (same vllm/vllm-openai:glm53-flash
// base, same overlay recipe) and already carries what the test-k3 images added
// as Stage-4 layers: FlashInfer 0.6.18 and the kpool indexer init are inside the
// residual, so neither flashinfer-0-6-18-stable nor glm53-indexer-init is listed.
//
// 3.1.0 is the 12.09 rebuild of 3.0.17 (processor_config.json resolved through
// the hub, gonka-ai/vllm#108; entrypoint no longer creates appuser,
// gonka-ai/gonka#1751) plus two layers that install gonka-poc v0.1.6. That
// plugin carries gonka-ai/gonka-vllm-plugins#10: the first PoC sequence of every
// batch landed on vLLM's null block and mamba-style kernels skipped it, so a
// miner's first-in-batch artifacts were unreproducible by any validator — the
// H200-vs-Blackwell invalid votes of 2026-09-14. The mlnode sources are
// unchanged from 3.0.17, so every Stage-4 layer below applies as before.
//
// Repinned 2026-09-16 to Vlad's third 3.1.0 (18:08 UTC): two layers over the
// first one. A per-model /versions cache in mlnode routes.py, since gonka-poc
// >= 0.1.6 reports poc_validation_inference per model and it is off for GLM;
// and serving.py without the replay max_tokens pin from kaitakuai/vllm#21
// (gonka-ai/vllm#111), which had hung two GLM nodes that morning. The
// effective serving.py is byte-identical to release/v0.28.0-glm53 after
// #111, so no replay layer is needed here.
//
// Tag 3.1.0-vllm-0.28.0 resolves to this digest; we pin by digest.
//
// Measured arm (B200): 4xB200 TP=4 (measured with the b300 test-k3 image): honest floor 0 of 1000 past the gate; batch 32 works once --max-num-batched-tokens allows it.
// Reports: kaitakuai/experiments/2026-09. Tracking: gonka-ai/gonka#1691.
b200_glm_5_3_flash: #OverlayProfile & bases.B200 & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "glm"
			model_revision: "5-3-flash"
		}
		version: {
			upstream: "3.1.0"
			rev:      2
		}
	}
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:de9150fcee0ad77199ca8a48ecae993b2cac0b92a7b05284d1575657d04522fa"
		upstream_version: "3.1.0"
	}
	// content-type-injector: patches/0001 is not in 3.1.0 (the sender-side fix, gonka#1756, is still open).
	// cold-start-tolerance: patches/0002, watcher grace for the 328 GiB load.
	// libnvrtc-symlink and sched-req-index-guard report no-op on this base
	// (merged as gonka#1560 and gonka-ai/vllm#106 respectively).
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"flashinfer-jit-uninstall",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
		"content-type-injector",
		"libnvrtc-symlink",
		"sched-req-index-guard",
		"poc-batch-size-force",
	]
	runner_patch: "b200-glm-5-3-flash-plugin"
	env: {
		MLNODE_VLLM_MODULE:                "gonka_poc.entrypoint.api_router"
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// 328 GiB of FP8 weights: a slow first load on every arm.
		VLLM_ENGINE_READY_TIMEOUT_S: "3600"
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
		// PoC batch for this hardware. The poc-batch-size-force layer above makes
		// it outrank whatever the caller sends; without that layer the value is
		// dead, since dapi omits batch_size and MLNode fills its own 32 in.
		// 32 is the batch the Blackwell measurements ran with, and what the
		// max-num-batched-tokens default above is sized for.
		POC_BATCH_SIZE_DEFAULT: "32"
	}
	runtime_defaults: {
		tensor_parallel_size: 4
		kv_cache_dtype:       "fp8"
		block_size:           2304
		max_num_seqs:         256
		max_num_batched_tokens: 65536
		logprobs_mode:        "processed_logprobs"
		trust_remote_code:    true
		tool_call_parser:     "glm47"
		reasoning_parser:     "glm45"
	}
	description: "B200 Blackwell SXM (x4) + GLM-5.3-Flash FP8 (TP=4, fp8 KV, autotune off) - vllm-poc 0.28 PLUGIN, overlay on gonka-ai/mlnode 3.1.0"
	notes: """
		Release-candidate image for GLM-5.3-Flash on Cortima's 3.1.0 base. The
		runner patch bakes the flags every measurement ran with; the chain
		proposal (gonka.gg #101) deliberately leaves block size, sequence cap and
		autotune out of the on-chain args so hosts can tune per hardware, which
		is what this image does for B200.
		"""
}
