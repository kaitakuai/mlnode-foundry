# Test fixtures

`upstream_runner.py.txt` — verbatim copy of
`mlnode/packages/api/src/api/inference/vllm/runner.py` from the gonka release
image our profiles overlay (currently `3.0.14-post2-vllm0.25.1-rc3`, tree
`13e7e3a`). `test_runner_patches.py` applies every referenced runner-patch to
it, which is the only place a patch's assumptions about upstream are checked
before a 10-minute image build.

**Refresh it whenever a profile's `base.digest` moves**, then run the tests:
a patch that no longer finds its anchor fails here instead of in Stage 4.
The `.txt` suffix keeps pytest and ruff from collecting it as a module.

`upstream_quant_config.py.txt` — verbatim `vllm/models/deepseek_v4/quant_config.py`
from gonka-ai/vllm `release/v0.25.1` (identical in the published rc3 image).
`patched_quant_config.py.txt` — the same file from kaitakuai/vllm#20's branch.
`test_runner_patches.py::test_dsv4_nvfp4_patch_matches_pr` applies our S4 patch
to the former and requires byte-equality with the latter. Refresh both when the
base image's vLLM moves or the PR is amended.
