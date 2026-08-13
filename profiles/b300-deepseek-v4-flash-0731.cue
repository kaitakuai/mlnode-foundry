// Profile: B300 + DeepSeek-V4-Flash-0731 FP8 (TP=1) — PRODUCTION, vLLM 0.25.1 line.
//
// Release-matrix leaf for the vLLM 0.25.1 update: overlay-mode on the
// release-branch mlnode base (exporter + hardened heartbeat in-tree,
// local patches 0001+0002 only). Model checkpoint and revision are pinned
// in tools/model-registry.cue (v4-flash-0731 @ 9e165c30); serving flags
// forced by the runner patch match the gonka-ai release configs
// (gonka#1536: 400k context, deepseek_v4 tokenizer/parsers) and the
// per-topology tuning validated in experiments
// 2026-08/deepseek-v4-flash-0731-1xb300.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b300_deepseek_v4_flash_0731: #OverlayProfile & bases.B300 & bases.DEEPSEEK_V4_FLASH & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "deepseek"
			model_revision: "v4-flash-0731"
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// rev=5 — DSpark draft experts off the NVFP4 path (kaitakuai/vllm#20).
			upstream: "3.0.16"
			rev:      5
		}
	}
	description: "B300 Blackwell Ultra SXM6 (x1, 8 engines/box) + DeepSeek-V4-Flash-0731 FP8 (TP=1, FlashMLA fp8 KV, 400k ctx) — vllm-poc 0.25.1 PLUGIN, release-matrix PRODUCTION image"
	// The chain pins deepseek-ai/…-0731, but B300 hosts run the NVFP4
	// conversion (gonka chat, 2026-08-10: "NVFP4 приняли технически как
	// основной вариант… из-за тотальной доминации"), and the card's throughput
	// figure is that variant's. Both checkpoints run on this image.
	advertised_model: {
		model:       "MJPansa/DeepSeek-V4-Flash-0731-NVFP4"
		model_short: "DeepSeek V4 Flash 0731 NVFP4"
	}
	mode: "upstream-overlay"
	base: {
		image:            "ghcr.io/gonka-ai/mlnode"
		// Cortima's PUBLISHED release image — see b300-kimi-k2-6.cue on the switch.
		digest:           "sha256:1b9b7ce55feecab837f1d7ce974fc5f377ae0a04a4fb403eeeb50130e7728ee1"
		upstream_version: "3.0.16"
	}
	// Plus the NVFP4 draft-MoE fix — B300-only by design: NVFP4 is the
	// technically-primary model variant on B300 alone (chat decision 2026-08-10).
	hw_patches: list.Concat([bases.GONKA_BASE_PATCHES, ["dsv4-nvfp4-draft-moe"]])
	runner_patch: "b300-deepseek-v4-flash-0731-plugin"
	env: {
		// Launch the gonka-poc composed entrypoint.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Worker extension collective_rpc channel.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
	}
}
