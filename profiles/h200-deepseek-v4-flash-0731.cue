// Profile: H200 + DeepSeek-V4-Flash-0731 FP8 (TP=2) — PRODUCTION, vLLM 0.25.1 line.
//
// Release-matrix leaf for the vLLM 0.25.1 update: overlay-mode on the
// release-branch mlnode base (exporter + hardened heartbeat in-tree,
// local patches 0001+0002 only). Model checkpoint and revision are pinned
// in tools/model-registry.cue (v4-flash-0731 @ 9e165c30); serving flags
// forced by the runner patch match the gonka-ai release configs
// (gonka#1536: 400k context, deepseek_v4 tokenizer/parsers) and the
// per-topology tuning validated in experiments
// 2026-08/deepseek-v4-flash-0731-2xh200.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

h200_deepseek_v4_flash_0731: #OverlayProfile & bases.H200 & bases.DEEPSEEK_V4_FLASH & {
	identity: {
		axes: {
			gpu:            "h200"
			model:          "deepseek"
			model_revision: "v4-flash-0731"
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// rev=3 — DSpark speculation forced on (Pasha, 2026-08-07).
			upstream: "3.0.14-post2-vllm0.25.1-rc3"
			rev:      3
		}
	}
	description: "H200 Hopper (x2) + DeepSeek-V4-Flash-0731 FP8 (TP=2, FlashMLA fp8 KV, 400k ctx) — vllm-poc 0.25.1 PLUGIN, release-matrix PRODUCTION image"
	mode: "upstream-overlay"
	base: {
		image: "ghcr.io/gonka-ai/mlnode"
		// MODE TEST (2026-08-07): base is cortima's PUBLISHED release image
		// ghcr.io/gonka-ai/mlnode:3.0.14-post2-vllm0.25.1-rc3 instead of our
		// own Stage 3. Proves the switch described in schema.cue: with
		// content-type-injector + cold-start-tolerance in hw_patches, this
		// yields the same functional image while dropping a whole build
		// stage and making mlnode drift impossible.
		digest:           "sha256:450983bbef31c8e19b8d24edb00c17520af7cc4fb0d186943f3ac3dec4dad387"
		upstream_version: "3.0.14-post2-vllm0.25.1-rc3"
	}
	hw_patches: bases.GONKA_MLNODE_PATCHES
	runner_patch: "h200-deepseek-v4-flash-0731-plugin"
	env: {
		// Launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Worker extension collective_rpc channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
	}
}
