// Profile base: what every overlay on cortima's published mlnode image needs.
// Imported by all `mode: "upstream-overlay"` leaves whose base is
// ghcr.io/gonka-ai/mlnode.
package bases

// Appended after the GPU base's own hw_patches. Each entry is a no-op when the
// base image already carries the equivalent change, so leaves concat this
// unconditionally instead of tracking which release candidate has what:
//
//   content-type-injector   our patches/0001 as a Stage-4 edit
//   cold-start-tolerance    our patches/0002 (watcher grace + runner timeout)
//   libnvrtc-symlink        -lnvrtc resolution; merged upstream as gonka#1560,
//                           so it reports "no-op" from rc4 on
//   sched-req-index-guard   kaitakuai/vllm#19 — EngineCore KeyError under async
//                           scheduling; patches vLLM, not mlnode, because the
//                           base image no longer comes from our residual tree
GONKA_BASE_PATCHES: [
	"content-type-injector",
	"cold-start-tolerance",
	"libnvrtc-symlink",
	"sched-req-index-guard",
]
