# mlnode-foundry

Build pipeline for Kaitaku-tuned mlnode container images.

> **Status:** Phase 1 (Discovery spike). One profile (`b300-kimi-int4`) builds
> locally end-to-end. Real Stage 3, CI, and full profile set land in PR #1-#3.
> See architecture in [`docs/architecture.md`](./docs/architecture.md) and
> [implementation epic](https://github.com/kaitakuai/mlnode-foundry/issues/7).

## Quickstart

```bash
# 1. Install tools
mise install                            # python 3.12 + cue 0.16

# 2. Install CLI in editable mode
pip install -e .

# 3. Validate a profile
mlnode-foundry validate b300-kimi-int4

# 4. Compute the tag this profile would produce
mlnode-foundry tag b300-kimi-int4
# → ghcr.io/kaitakuai/mlnode-b300-kimi:0.2.13-vllm0.20.0-q.int4-k1

# 5. Build locally (no push to GHCR)
mlnode-foundry build b300-kimi-int4

# 6. Verify ENV vars in the built image
docker run --rm ghcr.io/kaitakuai/mlnode-b300-kimi:0.2.13-vllm0.20.0-q.int4-k1 \
    sh -c 'env | grep VLLM'
```

## Repository layout

```
mlnode-foundry/
├── cue.mod/                  # Cue module manifest
├── tools/
│   └── naming.cue            # axes registry + naming policy (single source of truth)
├── profiles/
│   ├── _schema.cue           # #Profile = #BaseProfile | #OverlayProfile (sum type)
│   └── b300-kimi-int4.cue    # one profile for spike
├── mlnode_foundry/           # Python CLI (Typer)
│   ├── cli.py                # entrypoint
│   ├── cue.py                # subprocess wrapper for `cue` CLI
│   ├── render_bake.py        # profile + naming → buildx args + tag
│   └── validate.py           # `cue vet` runner
├── stage4/
│   └── Dockerfile            # parameterized via build-args
├── docs/
│   └── architecture.md       # full architecture spec
├── pyproject.toml            # Python project metadata
└── mise.toml                 # tool version pinning
```

## Boundary: Cue ↔ Python ↔ JSON

- **Cue** — human-authored intent: profiles, schemas, naming policy
- **Python** — orchestration: subprocess (cue, docker), file I/O, CLI
- **JSON** — machine-written observed state (added in Phase 3); validated by Cue schemas via `cue vet`

## License

Apache-2.0
