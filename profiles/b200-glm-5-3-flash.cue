package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

// GLM-5.3-Flash on Cortima's published mlnode 3.0.17, the first release image on
// the vLLM 0.28 residual (gonka-ai/vllm release/v0.28.0-glm53 after #104-#106,
// gonka-poc v0.1.4). Its S2 shares the first 32 layers with our validated
// kaitakuai/vllm-poc:glm53-poc-v4-ed8873884 (same vllm/vllm-openai:glm53-flash
// base, same overlay recipe) and already carries what the test-k3 images added
// as Stage-4 layers: FlashInfer 0.6.18 and the kpool indexer init are inside the
// residual, so neither flashinfer-0-6-18-stable nor glm53-indexer-init is listed.
// The mlnode sources in 3.0.17 are byte-identical to gonka-ai/gonka main.
//
// Tags 3.0.17, 3.0.17-glm53-h200 and 3.0.17-vllm-0.28.0-h200 resolve to the
// same digest; we pin the plain release tag.
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
			upstream: "3.0.17"
			rev:      1
		}
	}
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:b9ca935061bda3bd4f41f3906486bc419f4116df5f07134df76af14db21df7e1"
		upstream_version: "3.0.17"
	}
	// content-type-injector: patches/0001 is not in 3.0.17 (gonka#1590 still open).
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
	]
	runner_patch: "b200-glm-5-3-flash-plugin"
	env: {
		MLNODE_VLLM_MODULE:                "gonka_poc.entrypoint.api_router"
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// 328 GiB of FP8 weights: a slow first load on every arm.
		VLLM_ENGINE_READY_TIMEOUT_S: "3600"
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
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
	description: "B200 Blackwell SXM (x4) + GLM-5.3-Flash FP8 (TP=4, fp8 KV, autotune off) - vllm-poc 0.28 PLUGIN, overlay on gonka-ai/mlnode 3.0.17"
	notes: """
		Release-candidate image for GLM-5.3-Flash on Cortima's 3.0.17 base. The
		runner patch bakes the flags every measurement ran with; the chain
		proposal (gonka.gg #101) deliberately leaves block size, sequence cap and
		autotune out of the on-chain args so hosts can tune per hardware, which
		is what this image does for B200.
		"""
}
