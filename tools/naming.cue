// Package naming defines the axes registry and the global package/tag
// composition policy. Single source of truth for "how images are named."
//
// Profiles declare *which axes are set*; this file declares *how axes
// serialize* into a package name and tag string.
package tools

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
// New identity axis = add a stanza here + reference in profile spec.
axes: [string]: #Axis

// Name-axes (in package name, not in tag).
axes: gpu: {
	type:           "identity"
	name_axis_only: true
	description:    "GPU model (h100, h200, b200, b300, rtx6000, a100, ...)"
}
axes: model: {
	type:           "identity"
	name_axis_only: true
	description:    "Model family (qwen, kimi, minimax, deepseek, ...)"
}
axes: model_revision: {
	type:           "identity"
	name_axis_only: true
	description:    "Model revision (k26, v3-235b, m27, ...); cross-checked vs tools/model-registry.cue"
}

// Tag-axes (in tag, prefixed by .prefix).
axes: quant: {
	type:        "identity"
	prefix:      "q"
	description: "Quantization scheme"
	allowed_values: ["int4", "fp8", "nvfp4", "awq4bit", "mxfp4"]
}
axes: framework: {
	type:        "identity"
	prefix:      "f"
	description: "Inference engine (default: vllm, omitted from tag)"
	allowed_values: ["vllm", "sglang", "trtllm"]
	default: "vllm"
}
axes: build_mode: {
	type:        "identity"
	prefix:      "m"
	description: "Build mode — inference-only or inference+train"
	allowed_values: ["inference", "inference-train"]
	default: "inference"
}
axes: transform: {
	type:        "identity"
	prefix:      "t"
	status:      "reserved"  // not yet implemented; see ADR-0010
	description: "Post-build transform (full=untouched, slim=minimized via Stage 5)"
	allowed_values: ["full", "slim"]
	default: "full"
}

// Global naming policy — compose axes into package name and tag.
package: {
	prefix: "ghcr.io/kaitakuai/mlnode"
	axes: ["gpu", "model", "model_revision"]  // → mlnode-<gpu>-<model>-<model_revision>
}

tag: {
	// Identity-axes that appear in tag, in this order.
	// If a profile sets an axis to its default, the axis is omitted from tag
	// to keep tags short (e.g., framework=vllm doesn't get -f.vllm suffix).
	axes_order: ["quant", "framework", "build_mode", "transform"]
	modes: {
		"kaitakuai-base":   "{mlnode}-vllm{vllm}{tag_axes}-k{rev}"
		"upstream-overlay": "{upstream}-overlay{tag_axes}-k{rev}"
	}
}
