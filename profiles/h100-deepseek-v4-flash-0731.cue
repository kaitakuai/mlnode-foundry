// Profile: H100 + DeepSeek-V4-Flash-0731 FP8 (TP=4) — PRODUCTION, vLLM 0.25.1 line.
//
// Release-matrix leaf for the vLLM 0.25.1 update: overlay-mode on the
// release-branch mlnode base (exporter + hardened heartbeat in-tree,
// local patches 0001+0002 only). Model checkpoint and revision are pinned
// in tools/model-registry.cue (v4-flash-0731 @ 9e165c30); serving flags
// forced by the runner patch match the gonka-ai release configs
// (gonka#1536: 400k context, deepseek_v4 tokenizer/parsers) and the
// per-topology tuning validated in experiments
// 2026-08/deepseek-v4-flash-0731-dspark-4xh100.
package profiles

import "github.com/kaitakuai/mlnode-foundry/profiles/bases"

h100_deepseek_v4_flash_0731: #OverlayProfile & bases.H100 & bases.DEEPSEEK_V4_FLASH & {
	identity: {
		axes: {
			gpu:            "h100"
			model:          "deepseek"
			model_revision: "v4-flash-0731"
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// rev=2 — DSpark speculation forced on (Pasha, 2026-08-07).
			upstream: "3.0.14-post2-vllm0.25.1-rc3"
			rev:      2
		}
	}
	description: "H100 Hopper (x4) + DeepSeek-V4-Flash-0731 FP8 (TP=4, gmu 0.85, FlashMLA fp8 KV, 400k ctx) — vllm-poc 0.25.1 PLUGIN, release-matrix PRODUCTION image"
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		// Cortima's PUBLISHED release image — see b300-kimi-k2-6.cue on the switch.
		digest:           "sha256:450983bbef31c8e19b8d24edb00c17520af7cc4fb0d186943f3ac3dec4dad387"
		upstream_version: "3.0.14-post2-vllm0.25.1-rc3"
	}
	hw_patches: bases.GONKA_MLNODE_PATCHES
	runner_patch: "h100-deepseek-v4-flash-0731-plugin"
	env: {
		// Launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Worker extension collective_rpc channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// H100-only: 80 GiB/GPU against H200's 141. A PoC nonce reserves ~512
		// KV tokens, so the batch that fits scales with HBM — 16 here where the
		// rest of the 0731 matrix runs the default. Pasha, 2026-08-07.
		POC_BATCH_SIZE_DEFAULT: "16"
	}
}
