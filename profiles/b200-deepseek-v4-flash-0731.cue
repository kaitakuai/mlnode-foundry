// Profile: B200 + DeepSeek-V4-Flash-0731 FP8 (TP=2) — PRODUCTION, vLLM 0.25.1 line.
//
// Release-matrix leaf for the vLLM 0.25.1 update: overlay-mode on the
// release-branch mlnode base (exporter + hardened heartbeat in-tree,
// local patches 0001+0002 only). Model checkpoint and revision are pinned
// in tools/model-registry.cue (v4-flash-0731 @ 9e165c30); serving flags
// forced by the runner patch match the gonka-ai release configs
// (gonka#1536: 400k context, deepseek_v4 tokenizer/parsers) and the
// per-topology tuning validated in experiments
// 2026-08/deepseek-v4-flash-0731-2xb200.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

b200_deepseek_v4_flash_0731: #OverlayProfile & bases.B200 & bases.DEEPSEEK_V4_FLASH & {
	identity: {
		axes: {
			gpu:            "b200"
			model:          "deepseek"
			model_revision: "v4-flash-0731"
		}
		version: {
			// Overlay identity: upstream is our 0.25.1 release-line mlnode-base.
			// rev=1 — first production revision of the 0731 matrix.
			upstream: "3.0.14-post2-vllm0.25.1-rc1"
			rev:      1
		}
	}
	description: "B200 Blackwell (x2) + DeepSeek-V4-Flash-0731 FP8 (TP=2, FlashMLA fp8 KV, 400k ctx) — vllm-poc 0.25.1 PLUGIN, release-matrix PRODUCTION image"
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		// mlnode-base:0.2.14-vllm0.25.1-k5 — built from gonka-ai/gonka
		// vllm-0.25.1-upgrade @1b07e5c6 (exporter + hardened heartbeat carried
		// by the branch) over S2 vllm-poc:0.25.1 (gonka-ai/vllm release
		// @04a165c0 + gonka-vllm-plugins v0.1.1). Same base as the k10/k11
		// candidates the release was validated on.
		digest:           "sha256:7ba43ce4ad98d0d34c7b8626b424fffc2857c8dd4a2de86831e1b522fa09042b"
		upstream_version: "3.0.14-post2-vllm0.25.1-rc1"
	}
	// Ordered: mlnode source edits first (they are no-ops on an image built
	// from our own Stage 3, which already carries the equivalent patches),
	// then the link fix. This set is what makes mode "upstream-overlay"
	// produce the same image as "kaitakuai-base" -- see schema.cue.
	hw_patches: ["content-type-injector", "cold-start-tolerance", "libnvrtc-symlink"]
	runner_patch: "b200-deepseek-v4-flash-0731-plugin"
	env: {
		// Launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Worker extension collective_rpc channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
	}
}
