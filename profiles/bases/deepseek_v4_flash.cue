// Model base: DeepSeek-V4-Flash (DeepseekV4ForCausalLM, model_type deepseek_v4).
//
// Common (GPU-agnostic) env + dashboard-display runtime_defaults for every
// mlnode-<gpu>-deepseek-v4-flash leaf. Per the GLM_5_2 refactor lesson, the
// MoE/attention BACKEND env is GPU-specific and lives in each leaf — NOT here.
//
// V4 specifics that shape these defaults:
//   - --kv-cache-dtype fp8 is MANDATORY: the FlashMLA path coerces it to the
//     fp8_ds_mla layout and asserts otherwise. Surfaced as a display default;
//     each leaf's runner-patch forces it on the CLI.
//   - Compilation is auto-forced by vLLM to CompilationMode.NONE +
//     VLLM_USE_BREAKABLE_CUDAGRAPH=1 for deepseek_v4 (config/vllm.py). So we
//     display compilation_mode 0 / cudagraph_mode NONE — do NOT copy GLM's
//     mode 3. The PoC forward is eager via gonka-poc skip_compiled regardless;
//     never set VLLM_USE_BREAKABLE_CUDAGRAPH=0 (untested upstream).
//   - NO backend env here: V4 FP8 carries NATIVE ue8m0 scales (unlike GLM, no
//     float32-requant DeepGEMM linear crash → no VLLM_DISABLED_KERNELS), and
//     the default attention backend is deterministic FlashMLA-DSV4 — leaves
//     must NOT pin --attention-backend (FlashInfer-DSV4 has placeholder FP8
//     scales, consensus-unsafe).
package bases

DEEPSEEK_V4_FLASH: {
	env: {
		// Long 149 GiB FP8 load + MLA/sparse warmup.
		VLLM_RUNNER_TIMEOUT:         "3600"
		WATCHER_GRACE_FIRST_HEALTHY: "1"
		// Backend env intentionally omitted — GPU-specific, set per leaf.
	}
	runtime_defaults: {
		// Display only (each leaf's runner-patch forces the real CLI flags).
		// kv-cache-dtype fp8 is mandatory (FlashMLA fp8_ds_mla assert).
		kv_cache_dtype: *"fp8" | string
		// vLLM auto-forces NONE + breakable-cudagraph for deepseek_v4; the PoC
		// forward is eager via gonka-poc skip_compiled. Do NOT set mode 3.
		compilation_mode: *0 | int
		cudagraph_mode:   *"NONE" | string
	}
}
