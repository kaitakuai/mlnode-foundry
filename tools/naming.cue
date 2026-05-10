// Package naming defines the axes registry and the global package/tag
// composition policy. Single source of truth for "how images are named."
//
// Profiles only declare *which axes are set*; this file declares *how
// axes serialize* into a package name and tag string.
package naming

#AxisType: "identity" | "runtime"

#Axis: {
	type:            #AxisType
	description:     string
	prefix?:         string  // required for tag-axes (axes that appear in tag)
	name_axis_only?: bool    // true → only in package name, never in tag
	status?:         "active" | "reserved"
	default?:        string
	allowed_values?: [...string]
}

// Identity axes — change content of image (different image when value differs).
axes: [string]: #Axis

axes: gpu: {
	type:           "identity"
	name_axis_only: true
	description:    "GPU model (h100, b300, b200, rtx6000, a100)"
}
axes: model: {
	type:           "identity"
	name_axis_only: true
	description:    "Model family (qwen, kimi, minimax, deepseek, ...)"
}
axes: quant: {
	type:        "identity"
	prefix:      "q"
	description: "Quantization scheme (int4, fp8, nvfp4, awq4bit, ...)"
}

// Global naming policy — compose axes into package name and tag.
package: {
	prefix: "ghcr.io/kaitakuai/mlnode"
	axes: ["gpu", "model"]  // → mlnode-<gpu>-<model>
}

tag: {
	axes_order: ["quant"]  // identity-axes that appear in tag, in this order
	modes: {
		"kaitakuai-base":   "{mlnode}-vllm{vllm}{tag_axes}-k{rev}"
		"upstream-overlay": "{upstream}-overlay{tag_axes}-k{rev}"
	}
}
