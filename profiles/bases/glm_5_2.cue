// Profile base: GLM-5.2 FP8 common tunings.
// Imported by all glm+5-2 profiles.
package bases

GLM_5_2: {
	env: {
		// GLM-5.2 (GlmMoeDsaForCausalLM) DeepGEMM strategy on vLLM 0.23 — split the
		// two independent DeepGEMM flags so the MoE wins while the linear stays safe:
		//   - MoE experts ON DeepGEMM (the throughput lever; same path MiniMax/Kimi
		//     use). MoE-DeepGEMM is AND-gated on BOTH VLLM_USE_DEEP_GEMM=1 AND
		//     VLLM_MOE_USE_DEEP_GEMM=1 (fused_moe/oracle/fp8.py:360-361). Its
		//     PoC-forward workspace lock is handled by gonka-poc df73e1c
		//     (unlock/relock around the PoC forward — same fix as MiniMax).
		//   - LINEAR block-FP8 kernel routed to Cutlass via VLLM_DISABLED_KERNELS.
		//     GLM's MLA fused_qkv_a_proj is N=2624 (N%64==0 so the DeepGEMM linear
		//     kernel is SELECTED, but N%128==64 = partial last block); with E8M0
		//     requant on Blackwell this trips cudaErrorInvalidValue at memory
		//     profiling (Pasha's "image didn't start" crash). Disabling the two
		//     DeepGEMM/FlashInfer-DeepGEMM block-FP8 LINEAR kernels falls the
		//     selector through to CutlassFp8BlockScaledMMKernel (pads to 16-align,
		//     handles N%128!=0) — startup-safe, MoE keeps DeepGEMM+E8M0.
		// NONCE NOTE: routing the linear off DeepGEMM changes that layer's FP8
		// rounding (Cutlass float32 scales vs DeepGEMM UE8M0) — NOT nonce-bit-
		// identical to a pure-DeepGEMM validator. Acceptable while GLM is in the
		// benchmark phase (not governance-activated; no live L2 gate). Before GLM
		// goes to production mining, re-confirm fused_qkv_a_proj L2-equivalence
		// against the validator's kernel path.
		// FALLBACK if Fix A still won't start on GPU: drop VLLM_DISABLED_KERNELS and
		// set VLLM_USE_DEEP_GEMM_E8M0=0 (clears the crash but loses MoE E8M0 perf);
		// last resort triton-only: VLLM_USE_DEEP_GEMM=0 + VLLM_MOE_USE_DEEP_GEMM=0.
		VLLM_USE_DEEP_GEMM:          "1"
		VLLM_MOE_USE_DEEP_GEMM:      "1"
		VLLM_USE_DEEP_GEMM_E8M0:     "1"
		VLLM_USE_FLASHINFER_MOE_FP8: "0"
		VLLM_DISABLED_KERNELS:       "DeepGemmFp8BlockScaledMMKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
		// 753B FP8 (~ full 8×B200) load + DSA warmup is slow — long runner
		// timeout + first-healthy grace.
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
	}
	runtime_defaults: {
		// Inference runs COMPILED by default (vLLM CompilationMode.VLLM_COMPILE +
		// CUDA graphs) for decode throughput — on GLM that is 817 vs 157 tok/s
		// (+6.8× TPOT) over eager. The PoC forward runs EAGER on its own via
		// gonka-poc poc_model_runner (set_forward_context skip_compiled=True), so
		// one image gives eager PoC (bit-compat, ~+25% nonces) + compiled inference.
		// NOT forced by the runner-patch (matches b300-minimax) — these are the
		// effective vLLM defaults, shown for the dashboard. For the EXCEPTIONAL
		// case where inference must also be eager, the operator passes
		// --enforce-eager at launch (the patch neither forces nor strips it).
		compilation_mode: *3 | int           // VLLM_COMPILE (compiled inference)
		cudagraph_mode:   *"PIECEWISE" | string // vLLM compiled default; not forced
	}
}
