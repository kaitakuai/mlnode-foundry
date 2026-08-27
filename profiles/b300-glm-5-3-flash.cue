// Profile: B300 Blackwell Ultra (×2) + GLM-5.3-Flash FP8 — vllm-poc PLUGIN, vLLM 0.28,
// UPSTREAM-TEST mode. Bring-up image for the model.
//
// Built FROM an explicit mlnode-base digest rather than tools/stage3.lock.cue,
// so the fleet's shared lock stays on 0.25.1 while this leaf rides the one-off
// 0.28 chain: kaitakuai/vllm@poc-residual-glm53 -> vllm-poc:glm53-poc-v4 ->
// mlnode-base:0.2.14-vllm0.28-glm53-k2.
//
// The model is not in any upstream vLLM release: support lives in the open PR
// vllm-project/vllm#53906, and because that diff reaches into .cu and cmake the
// kernels have to come pre-built from vllm/vllm-openai:glm53-flash. Provenance
// and the byte-for-byte comparison that justifies overlaying our residual onto
// that image are in PORTING.md on the residual branch.
//
// Serving config mirrors Crash_Bash_FL's bring-up command of 2026-08-27, the
// only configuration this checkpoint has been started with. TP=2 because 328
// GiB of FP8 weights fit in 2×275 GiB. fp8 KV cache as in his run. NOTHING here
// is benchmarked: no batch sizing, no context cap, no backend or compilation
// pins — a wrong pin on an unmeasured model is a consensus hazard, not a slow
// path.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b300_glm_5_3_flash: #OverlayProfile & bases.B300 & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "glm"
			model_revision: "5-3-flash"
		}
		version: {
			// "upstream" here is OUR 0.28 mlnode-base, not a product-science release.
			upstream: "0.2.14-vllm0.28-glm53"
			rev:      1
		}
	}
	mode: "upstream-test"
	base: {
		image: "ghcr.io/kaitakuai/mlnode-base"
		// 0.2.14-vllm0.28-glm53-k2 (run 33000436174) from S2
		// vllm-poc:glm53-poc-v4-ed8873884 @sha256:31b42acc — the residual carries
		// the canonical 8-commit stack plus the two fixes we used to ship as S4
		// layers: the scheduler guard (kaitakuai/vllm#19) and all of #21. #21 covers
		// the V1 sampling path; this checkpoint actually runs on V2, where the
		// replay hooks already live, so it is insurance here rather than a
		// prerequisite — see PORTING.md on the residual branch.
		digest:           "sha256:602b13befec31d38a842f6f6eb60e0ed74afb4a4b31bf823c09cca947cc33c20"
		upstream_version: "0.2.14-vllm0.28-glm53"
	}
	// The bases.B300 set, spelled out because ORDER matters here: the FlashInfer
	// bump Crash_Bash_FL asked for has to land BEFORE flashinfer-jit-uninstall,
	// not after. The bump installs all three distributions at one version — they
	// are checked against each other at import — and the uninstall then drops the
	// jit-cache again, which is the point on sm_103a: the prebuilt cache targets
	// another SM, so JIT-compiling for the real GPU beats shipping dead kernels.
	// Reversed, the uninstall would run first and the bump would put the cache
	// straight back.
	//
	// The two Stage-4 layers this line used to carry (sched-req-index-guard,
	// content-type-injector) are already inside: the first in the residual, the
	// second via the Stage-3 patches/0001.
	hw_patches: [
		"triton-ptxas-from-system-cuda",
		"libcuda-compat-580-driver",
		"nvidia-headers-symlinks",
		"cold-start-tolerance",
		"flashinfer-0-6-18-nightly",
		"flashinfer-jit-uninstall",
	]
	runner_patch: "b300-glm-5-3-flash-plugin"
	env: {
		// Server-side plugin flip: launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Required for the worker extension's collective_rpc msgpack channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// 328 GiB across 2 ranks is a slow first load; his bring-up command sets
		// the same 3600s, so an engine that is merely loading is not killed.
		// VLLM_RUNNER_TIMEOUT / WATCHER_GRACE_FIRST_HEALTHY come from bases.B300.
		VLLM_ENGINE_READY_TIMEOUT_S: "3600"
	}
	runtime_defaults: {
		// Forced via the runner-patch; reproduced here for the dashboard.
		tensor_parallel_size: 2
		kv_cache_dtype:       "fp8"
		logprobs_mode:        "processed_logprobs"
		trust_remote_code:    true
		tool_call_parser:     "glm47"
		reasoning_parser:     "glm45"
	}
	description: "B300 Blackwell Ultra SXM6 (×2) + GLM-5.3-Flash FP8 (TP=2, fp8 KV) — vllm-poc 0.28 PLUGIN, test-mode image"
	notes: """
		TEST bring-up image, built at Crash_Bash_FL's request (2026-08-27) as the
		Blackwell arm of a two-image pair; the Hopper arm is h100-glm-5-3-flash,
		which disables FlashInfer autotune instead of pinning fp8 KV.

		Nothing about this model is measured yet. It has no chain governance
		record, its vLLM support is an open upstream PR, and the base image it
		rides was built from an untagged tree (see PORTING.md). Do not treat any
		number produced by this image as a reference until a GPU pass covers
		self-validation, nonce/min and a replay cross-check.

		The 0.28 chain is a side branch: tools/stage3.lock.cue still points the
		fleet at the 0.25.1 stack, and this leaf reaches its base by digest.
		"""
}
