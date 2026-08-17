// Profile: B300 Blackwell Ultra + MiniMax-M2.7 FP8 (TP=1, single-GPU) — PLUGIN base.
//
// Composition: #BaseProfile & B300 & MINIMAX_M2_7 with TP=1, built on the
// vllm-poc PLUGIN base (residual vLLM 0.23 + gonka-poc package: worker
// extension + composed entrypoint; see ADR-0013), NOT the legacy fat-fork
// monolith. This profile was MIGRATED IN PLACE from the fat-fork vLLM 0.20
// monolith to the 0.23 plugin base (per the decision to retire the fat-fork
// build line — foundry stops BUILDING fat-fork; the previously published
// fat-fork image 0.2.13-vllm0.20.0-k1 persists on GHCR as a redeploy rollback).
//
// Tag effect of the migration: vllm 0.20.0 → 0.23.0. The vllm version is part
// of the tag template ({mlnode}-vllm{vllm}{tag_axes}-k{rev}), so this profile
// now renders 0.2.13-vllm0.23.0-k1 (vs the retired 0.2.13-vllm0.20.0-k1). The
// filename/axes stay b300-minimax-m2-7 (1:1 with the image package name).
//
// Key plugin differences vs the retired fat-fork b300-minimax profile:
//   - hw_patches drops `poc-householder-compile`: the householder torch.compile
//     wrap lived in the fat-fork's vllm/poc/ tree; on the plugin base the PoC
//     math (and any compile wrapping) ships inside the gonka-poc package, so
//     the Stage-4 fragment that edited vllm/poc/gpu_random.py no longer applies.
//   - env adds MLNODE_VLLM_MODULE so the mlnode runner launches the composed
//     gonka-poc entrypoint instead of vLLM's stock api_server.
//   - runner_patch injects the plugin-specific forced engine args (worker
//     extension class, processed logprobs, FLASHINFER backend). It does NOT
//     force --enforce-eager — PoC eager is handled by gonka-poc's skip_compiled.
//
// Throughput/quality claims below are INHERITED from the fat-fork validation
// (1×B300 SXM6, 2026-05-23) and are NOT yet re-validated on the 0.23 plugin
// base — see the validation-report tuning_note severity and blockers.
// 0.25.1 MIGRATION NOTE (2026-08-06): base flipped to the release-line
// mlnode-base 0.2.14-vllm0.25.1-k5 via tools/stage3.lock.cue; serving flags
// are INHERITED from the vLLM 0.20 fat-fork validation campaigns
// (experiments 2026-05/minimax-m27-*) and are NOT yet revalidated on the
// 0.25.1 plugin stack. Backend selection is the sensitive part (TRTLLM
// auto-select on Blackwell, forced triton on Hopper, marlin on A100 —
// consensus-relevant, see the Marlin/DeepGEMM cross-hw precedent). Treat
// these images as CANDIDATES until a hardware pass equivalent to the
// deepseek 2026-08 campaigns is run.
package profiles

import (
	"list"

	"github.com/kaitakuai/mlnode-foundry/profiles/bases"
)

b300_minimax_m2_7: #OverlayProfile & bases.B300 & bases.MINIMAX_M2_7 & {
	identity: {
		axes: {
			gpu:            "b300"
			model:          "minimax"
			model_revision: "m2-7"
			// Test-line marker: renders the tag as 3.0.16-overlay-poc.decode-k1
			// — a separate tag line from the production prefill k-series, so a
			// decode candidate can never be mistaken for the next prod rev.
			poc: "decode"
		}
		version: {
			// Overlay identity: upstream is cortima's published mlnode image.
			// poc=decode line, rev=1 — DECODE-PoC CANDIDATE: the gonka-poc package is swapped to
			// the decode branch (feat/decode-poc-v2 @ c0a19daf, hw-validated
			// campaign 2026-08-15/16). The decode branch DROPS the prefill
			// scheme: this image CANNOT serve the current network's prefill PoC
			// and must not be promoted before the coordinated consensus switch.
			// The prefill k-line (…-overlay-k6) stays untouched as production.
			// rev=6 — governance defaults added when DAPI has not broadcast
			// them; same policy change as b200 rev=6 (Pasha, 2026-08-14).
			upstream: "3.0.16"
			rev:      3
		}
	}
	mode: "upstream-overlay"
	base: {
		// Cortima's PUBLISHED release image — their Stage-3 equivalent
		// (mlnode/packages/api/Dockerfile over ghcr.io/gonka-ai/vllm:v0.25.1-poc-v2).
		// Taking it as our base drops a whole build stage and makes drifting
		// from their mlnode source impossible; everything we still add lives
		// in hw_patches + runner_patch below. See schema.cue on the switch.
		image:            "ghcr.io/gonka-ai/mlnode"
		digest:           "sha256:1b9b7ce55feecab837f1d7ce974fc5f377ae0a04a4fb403eeeb50130e7728ee1"
		upstream_version: "3.0.16"
	}
	// The fat-fork's `poc-householder-compile` fragment is INTENTIONALLY absent:
	// it edited vllm/poc/gpu_random.py in the monolith, which does not exist as
	// a Stage-4-patchable tree on the plugin base (PoC math lives in gonka-poc).
	// We DE-REFERENCE only — the shared fragment file stays for the 5
	// non-migrated profiles.
	// decode-poc-plugin LAST: the fragments before it patch the base image;
	// the plugin swap replaces gonka-poc wholesale and must not be overwritten.
	hw_patches: list.Concat([bases.B300.hw_patches, bases.GONKA_BASE_PATCHES, ["decode-poc-plugin", "pow-v2-decode-fields"]])
	runner_patch: "b300-minimax-decode-plugin"
	env: {
		// Tell the mlnode runner to launch the gonka-poc COMPOSED entrypoint
		// (stock build_app + PoC router + gating middleware) instead of vLLM's
		// stock api_server. This is the server-side half of the plugin flip
		// (ADR-0013); the worker-side half is --worker-extension-cls, forced by
		// the runner-patch.
		MLNODE_VLLM_MODULE: "gonka_poc.entrypoint.api_router"
		// Belt-and-suspenders: the vllm-poc S2 base already sets this, but the
		// gonka-poc worker extension is loaded via collective_rpc serialization,
		// which requires insecure serialization to be allowed. Pin it at the
		// profile level too so the image is correct even if the base ever drops it.
		VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
		// Slow MiniMax-M2.7 cold start (FP8 KV + 180000 max-model-len) needs the
		// long runner timeout + first-healthy grace; inherited from the B300 /
		// MINIMAX_M2_7 bases, reproduced here for clarity.
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
		// Decode-PoC runtime: capture mode is the release execution mode, and
		// the chunk clamp holds the decode batch at the measured KV wall for
		// gmu 0.95 (536 nonces × 512 tokens ≤ 278896-token KV pool). The lease
		// is taken for the CLIENT's batch_size, so an uncapped client batch
		// above the wall pays a measured 6-7 s lease timeout per chunk.
		POC_DECODE_CAPTURE:   "1"
		POC_DECODE_MAX_BATCH: "536"
		// Mining round batch: mlnode omits batch_size (decode pass-through
		// patch), so this backend default decides the round. 536 = the
		// measured KV wall for MiniMax on B300 at gmu 0.95; a smaller round
		// wastes the wall (32 would run ~7 n/s instead of 30).
		POC_BATCH_SIZE_DEFAULT: "536"
	}
	runtime_defaults: {
		// 1 × B300 SXM6 = 275 GiB HBM. MiniMax-M2.7 FP8 (230 GB) fits with
		// 1.23× concurrency at max_model_len=180000 (KV pool 221616 tokens,
		// GPU ~91% full). Highest per-GPU PoC throughput of any tested
		// hardware on the fat-fork base (1792 nonces/min/GPU) — plugin-base
		// throughput not yet re-measured.
		tensor_parallel_size: 1
		// Attention backend pinned to FLASHINFER explicitly (also forced via the
		// runner-patch) so the auto-selector heuristic doesn't silently regress
		// this profile across vLLM releases.
		attention_backend: "FLASHINFER"
	}
	description: "B300 Blackwell Ultra SXM6 (1×) + MiniMax-M2.7 FP8 (TP=1) — DECODE-PoC CANDIDATE (plugin feat/decode-poc-v2 @ c0a19daf; not for the prefill network)"
	notes: """
		In-place plugin migration of the b300-minimax-m2-7 profile (was fat-fork
		vLLM 0.20). Built on the vllm-poc 0.23 base (residual vLLM + gonka-poc
		package) per ADR-0013, so PoC math ships as an out-of-tree plugin instead
		of a forked vllm/poc/ tree.

		Plugin flip (vs the retired fat-fork b300-minimax):
		  - MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router (composed
		    server: stock build_app + PoC router + gating middleware).
		  - runner-patch forces --worker-extension-cls gonka_poc.worker.PoCWorkerExtension
		    (worker-side PoC execution via public collective_rpc), plus
		    --logprobs-mode processed_logprobs and --attention-backend FLASHINFER.
		    It does NOT force --enforce-eager: the PoC forward is already eager via
		    gonka_poc.poc.poc_model_runner (skip_compiled=True), so inference keeps
		    CUDA graphs (no throughput regression) while PoC stays bit-compatible.
		  - poc-householder-compile hw-patch DROPPED — the householder
		    torch.compile wrap targeted the monolith's vllm/poc/gpu_random.py,
		    which is not a Stage-4-patchable file on the plugin base.

		Chain-governance MiniMax args (--max-model-len 180000, --kv-cache-dtype
		fp8, --tool-call-parser minimax_m2, --reasoning-parser
		minimax_m2_append_think, --enable-auto-tool-choice) are NOT baked by the
		runner-patch: they continue to flow from the network-node DAPI broadcast
		into mlnode runner.py self.additional_args and reach the composed
		gonka-poc engine unchanged (verified: the runner assembles additional_args
		identically regardless of which entrypoint module it launches).

		Throughput/quality figures inherited from the fat-fork validation
		(1×B300 SXM6, 2026-05-23): 1792 nonces/min @ batch=32; nonces
		cross-validate with the 2×B200 baseline (mean L2 0.266, PASS under the
		MiniMax chain gate 0.75/0.10). These are NOT yet re-validated on the 0.23
		plugin base; re-benchmark on B300 hardware before this image is
		considered production-validated. CUDA note: the residual S1 bases on
		vLLM's recommended default image (vllm/vllm-openai:v0.23.0 → CUDA 13.0.2),
		matching the validated fat-fork 0.20 B300 image and the shared
		stage3.lock cuda field.
		"""
	tuning_notes: [
		{
			knob:     "decode-poc-campaign-2026-08"
			source:   "migration-025-kit (Google Drive, gonka-poc-migration-025-2026-08-16): PROFILES.md + RESULTS_FINAL.json"
			reason:   "DECODE launch profile hardware-validated on 1×B300 SXM6 (2026-08-15/16): PoC 30.36 nonces/s @ batch 536 (=1822/min), chat 30.00 req/s @ c536, R=1.012; gmu 0.95 + mns 704 + capture 608 + FLASHINFER + flashinfer_trtllm. Self-validation 0 at TP=1; 0.20-golden match under the consensus rule. Thresholds NOT final (tau/p_mismatch pending the binomial-test context) — validation defaults in code produce false fraud cross-hardware."
			severity: "warning"
			added_at: "2026-08-17"
		},
		{
			knob:     "validation-report"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Hardware validation INHERITED from the fat-fork b300-minimax profile (1×B300 SXM6, 275 GiB HBM, 2026-05-23): PoC 1792 nonces/min @ batch=32, nonces cross-validate with 2×B200 baseline (mean L2 0.266, PASS under MiniMax chain gate 0.75/0.10). NOT yet re-validated on the vllm-poc 0.23 PLUGIN base — re-benchmark on B300 hardware before treating as production-validated."
			severity: "warning"
			added_at: "2026-06-21"
		},
		{
			knob:     "MLNODE_VLLM_MODULE=gonka_poc.entrypoint.api_router"
			source:   "docs/adr/0013-poc-integration-architecture.md (server-side composed entrypoint)"
			reason:   "Server-side half of the fork→plugin flip: launches the gonka-poc composed entrypoint (stock vLLM build_app + PoC router + gating middleware) instead of vLLM's stock api_server. Required for the plugin base — the monolith's in-tree PoC router is gone. No throughput trade-off; info."
			added_at: "2026-06-21"
		},
		{
			knob:     "poc-householder-compile=dropped"
			source:   "docs/adr/0013-poc-integration-architecture.md + tools/hw-patches/poc-householder-compile.dockerfile header"
			reason:   "The householder torch.compile wrap (fat-fork +10-12% on Qwen3-235B-FP8) edited the monolith's vllm/poc/gpu_random.py. On the plugin base, PoC math ships inside the gonka-poc package, so that Stage-4 fragment no longer has a file to patch and is de-referenced. Any equivalent compile wrap is now a gonka-poc-internal concern, not a foundry hw-patch. Potential perf delta unmeasured on the plugin base — info, not a baseline-below warning."
			added_at: "2026-06-21"
		},
		{
			knob:     "attention_backend=FLASHINFER"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "vLLM auto-selector picks FLASHINFER on Blackwell; pinned explicitly (and forced via the runner-patch) so the auto-selector heuristic doesn't silently regress this profile across vLLM versions. No baseline violation; info."
			added_at: "2026-06-21"
		},
		{
			knob:     "max_model_len=180000"
			source:   "gonka-ai/gonka:inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel"
			reason:   "Chain governance mandates 180000; experiments validated KV-pool fit (221616 tokens, 1.23× concurrency at 180000) on B300 fat-fork, but generation quality at full 180k context not specifically swept, and not re-checked on the 0.23 plugin base. Flows from DAPI broadcast, not the runner-patch. Lower would diverge on long-context prompts; below-native-context = warning."
			severity: "warning"
			added_at: "2026-06-21"
		},
		{
			knob:     "tensor_parallel_size=1"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Validated single-B300 (275 GiB HBM) configuration on the fat-fork base — MiniMax-M2.7 FP8 (230 GB) fits with 1.23× concurrency at max_model_len=180000. Phase 3 batch sweep on B300: 8→1392, 16→1728, 32→1792 (best), 64→hang. The chain VRam=320 GB nominal requirement is conservative; 1×B300 cross-validates with the B200 fleet under the MiniMax chain gate."
			added_at: "2026-06-21"
		},
		{
			knob:     "max_num_seqs=128 (batch_size=64 hard ceiling)"
			source:   "https://github.com/kaitakuai/experiments/blob/main/2026-05/minimax_m27_1xb300_sxm6/README.md"
			reason:   "Phase 3 sweep on B300 (fat-fork) confirms batch_size=64 hangs the PoC engine (OOM-stuck, not crash — engine never recovers). Same failure mode as B200/H200/H100/A100 MiniMax profiles. max_num_seqs=128 from MINIMAX_M2_7 base is safe because vLLM runtime caps actual batch by KV cache; the hazard is operator-supplied PoC batch override. Not re-swept on the plugin base."
			severity: "warning"
			added_at: "2026-06-21"
		},
	]
}
