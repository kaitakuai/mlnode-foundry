package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

// GLM-5.3-Flash on Cortima's published mlnode 3.0.17, the first release image on
// the vLLM 0.28 residual (gonka-ai/vllm release/v0.28.0-glm53 after #104-#106,
// gonka-poc v0.1.4). Its S2 shares the first 32 layers with our validated
// kaitakuai/vllm-poc:glm53-poc-v4-ed8873884 (same vllm/vllm-openai:glm53-flash
// base, same overlay recipe) and already carries what the test-k3 images added
// as Stage-4 layers: FlashInfer 0.6.18 and the kpool indexer init are inside the
// residual, so neither flashinfer-0-6-18-stable nor glm53-indexer-init is listed.
//
// Rebuilt 2026-09-12 on the digest below, which carries two fixes the first
// 3.0.17 lacked: the GLM processor resolves processor_config.json through the
// hub (gonka-ai/vllm#108), so `--model <hf id>` works — the path MLNode takes,
// since it downloads the checkpoint but passes the id — and the entrypoint no
// longer creates appuser at runtime (gonka-ai/gonka#1751), which retires the
// drop-stock-ubuntu-user layer. Its mlnode sources are the release branch
// feat/glm-5-3-flash-release, i.e. main plus that entrypoint fix.
//
// Tags 3.0.17, 3.0.17-vllm-0.28.0, 3.0.17-vllm-0.28.0-h200 and
// 3.0.17-glm53-h200 all resolve to this digest; we pin by digest.
//
// Measured arm (H100): 8xH100 TP=8: 1775 nonces/min at batch 16; needs --gpu-memory-utilization 0.95 to start on 80 GB cards, which also caps PoC batch at 16 (32 is OOM).
// Reports: kaitakuai/experiments/2026-09. Tracking: gonka-ai/gonka#1691.
h100_glm_5_3_flash: #OverlayProfile & bases.H100 & {
	identity: {
		axes: {
			gpu:            "h100"
			model:          "glm"
			model_revision: "5-3-flash"
		}
		version: {
			upstream: "3.0.17"
			rev:      5
		}
	}
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:6772abdf736bbe8cad27d8c305e1fa32b54c82f783286d405fc5171d06419081"
		upstream_version: "3.0.17"
	}
	// content-type-injector: patches/0001 is not in 3.0.17 (gonka#1590 still open).
	// cold-start-tolerance: patches/0002, watcher grace for the 328 GiB load.
	// libnvrtc-symlink and sched-req-index-guard report no-op on this base
	// (merged as gonka#1560 and gonka-ai/vllm#106 respectively).
	// GONKA_BASE_PATCHES minus drop-stock-ubuntu-user: this base carries
	// gonka#1751, so the entrypoint no longer creates appuser and the stock
	// account is harmless. The shared list keeps it for bases built earlier.
	hw_patches: [
		"content-type-injector",
		"cold-start-tolerance",
		"libnvrtc-symlink",
		"sched-req-index-guard",
		"poc-batch-size-from-env",
	]
	runner_patch: "h100-glm-5-3-flash-plugin"
	env: {
		MLNODE_VLLM_MODULE:                "gonka_poc.entrypoint.api_router"
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// 328 GiB of FP8 weights: a slow first load on every arm.
		VLLM_ENGINE_READY_TIMEOUT_S: "3600"
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
		// PoC batch when the request carries none, which is the production path:
		// dapi omits batch_size on purpose. Needs the poc-batch-size-from-env
		// layer above, or MLNode's own default of 32 wins. 16 is the usable PoC batch on Hopper: 32 fails in the FlashInfer sparse-MLA
		// kernel on H200 and is OOM on 8xH100.
		// Crash_Bash_FL, 2026-09-10.
		POC_BATCH_SIZE_DEFAULT: "16"
	}
	runtime_defaults: {
		tensor_parallel_size: 8
		kv_cache_dtype:       "fp8"
		block_size:           2304
		max_num_seqs:         256
		gpu_memory_utilization:  0.95
		max_num_batched_tokens: 65536
		logprobs_mode:        "processed_logprobs"
		trust_remote_code:    true
		tool_call_parser:     "glm47"
		reasoning_parser:     "glm45"
	}
	description: "H100 Hopper SXM5 (x8) + GLM-5.3-Flash FP8 (TP=8, fp8 KV, autotune off, gmu 0.95) - vllm-poc 0.28 PLUGIN, overlay on gonka-ai/mlnode 3.0.17"
	notes: """
		Release-candidate image for GLM-5.3-Flash on Cortima's 3.0.17 base. The
		runner patch bakes the flags every measurement ran with; the chain
		proposal (gonka.gg #101) deliberately leaves block size, sequence cap and
		autotune out of the on-chain args so hosts can tune per hardware, which
		is what this image does for H100.
		"""
}
