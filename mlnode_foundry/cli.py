"""mlnode-foundry CLI — Typer entrypoint."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from .build_hash import compute_profile_hash
from .paths import REPO_ROOT
from .profile_factory import DEFAULT_MLNODE_VERSION, DEFAULT_VLLM_VERSION
from .render_bake import load_profile, render_build_args
from .render_name_tag import load_naming, render_package_and_tag
from .runner import list_runners, select_runner
from .validate import validate_profile

app = typer.Typer(
    help="mlnode-foundry: build pipeline for Kaitaku-tuned mlnode container images.",
    no_args_is_help=True,
)
profile_app = typer.Typer(help="Profile authoring commands.", no_args_is_help=True)
runner_app = typer.Typer(help="Runner inventory commands.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")
app.add_typer(runner_app, name="runner")


# --- Top-level commands ------------------------------------------------------


@app.command()
def validate(profile: str) -> None:
    """Validate a profile against the schema (cue vet)."""
    validate_profile(profile)
    typer.secho(f"✓ Profile '{profile}' is valid.", fg=typer.colors.GREEN)


@app.command()
def tag(profile: str) -> None:
    """Print the computed package:tag for a profile."""
    p = load_profile(profile)
    naming = load_naming()
    pkg, t = render_package_and_tag(p, naming)
    typer.echo(f"{pkg}:{t}")


@app.command()
def hash(profile: str) -> None:  # noqa: A001 — shadows builtin intentionally as CLI verb
    """Print the content hash of all inputs that affect this profile's build."""
    h = compute_profile_hash(profile)
    typer.echo(h)


@app.command()
def expand(profile: str) -> None:
    """Print self-contained resolved profile (all bases inlined) as JSON.

    Useful for audit (see exactly what `cue export` resolved a profile to),
    sharing (send one JSON instead of the whole repo), and debugging tag
    rendering (what axes are actually set?).
    """
    resolved = load_profile(profile)
    typer.echo(json.dumps(resolved, indent=2))


@app.command("image-size")
def image_size_cmd(ref: str) -> None:
    """Print compressed image size for linux/amd64 (e.g. '15 GB'), empty if unknown."""
    from .image_size import fetch_image_compressed_size, humanize_bytes

    n = fetch_image_compressed_size(ref)
    # Suppress "0 B" so the workflow + renderer fall back to null
    # (avoids registry-view `"size": "0 GB"` placeholder lies).
    if n > 0:
        typer.echo(humanize_bytes(n))


def _preserve_hand_edited_metrics(view: dict, output_path: Path) -> dict:
    """Carry forward hand-edited nonces/weight from an existing registry-view JSON.

    Plan B benchmark workflow:
      1. CI builds the image (this command runs; view["nonces"] = view["weight"] = None).
      2. Operator tests the image on real hardware, measures throughput.
      3. Operator opens a small PR editing only `registry-view/<file>.json` to set
         `"nonces": <8-GPU-normalized-count>` (no profile change, no rebuild).
      4. Some later commit triggers Stage 4 rebuild of this profile. This
         function reads the EXISTING registry-view JSON before overwriting,
         finds the hand-edited number, and carries it forward into the
         fresh render — so the rebuild does NOT regress the dashboard chip
         back to "no measurement".

    Validation is strict:
      - existing value MUST be int/float >= 0 (silently ignored otherwise)
      - already-set view fields (from explicit CLI args, future use) win

    Failure modes (corrupt JSON, IO error) → silent fallback to nulls. The
    rebuild still succeeds; the operator just needs to re-set the number.
    """
    if not output_path.exists():
        return view
    try:
        existing = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return view
    # report_url joins nonces/weight: the renderer emits null when the profile
    # has no tuning_note pointing at a report, and null means "no information",
    # never "erase what the operator recorded".
    for field in ("nonces", "weight", "report_url"):
        existing_val = existing.get(field)
        if existing_val is None:
            continue
        if field == "report_url":
            if not isinstance(existing_val, str) or not existing_val.startswith("http"):
                continue
        else:
            if not isinstance(existing_val, int | float) or isinstance(existing_val, bool):
                continue
            if existing_val < 0:
                continue
        if view.get(field) is not None:
            continue  # explicit caller value beats preservation
        view[field] = existing_val
    return view


@app.command("registry-view")
def registry_view_cmd(
    profile: str,
    digest: str = typer.Option("", help="Image digest (sha256:...); empty for local preview"),
    cosign_identity: str = typer.Option("", help="Cosign signing identity (workflow URL)"),
    size: str = typer.Option("", help="Humanized image size, e.g. '15 GB'"),
    output: Path = typer.Option(  # noqa: B008
        Path(""),
        help="Output JSON path. Defaults to registry-view/<package-basename>-<tag>.json",
    ),
) -> None:
    """Render dashboard-compatible registry-view JSON for a profile."""
    from .render_registry_view import render_registry_view

    view = render_registry_view(
        profile,
        digest=digest or None,
        cosign_identity=cosign_identity or None,
        size=size or None,
    )
    if str(output) == "" or str(output) == ".":
        package_basename = view["name"].rsplit("/", 1)[-1]
        output = REPO_ROOT / "registry-view" / f"{package_basename}-{view['tag']}.json"
    view = _preserve_hand_edited_metrics(view, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(view, indent=2) + "\n")
    typer.secho(f"✓ Wrote: {output}", fg=typer.colors.GREEN)


@app.command("render-dockerfile")
def render_dockerfile_cmd(
    profile: str,
    output: Path = typer.Option(  # noqa: B008
        REPO_ROOT / "stage4" / "Dockerfile.rendered",
        help="Where to write the rendered Dockerfile.",
    ),
) -> None:
    """Render stage4/Dockerfile.tmpl for a profile (substitute ENV block + tuning label)."""
    from .render_dockerfile import render_dockerfile

    p = load_profile(profile)
    path = render_dockerfile(p, output)
    typer.secho(f"✓ Rendered: {path}", fg=typer.colors.GREEN)


@app.command()
def build(
    profile: str,
    push: bool = typer.Option(
        False, "--push/--no-push", help="Push to registry after build (default: no push)."
    ),
) -> None:
    """Build a profile via docker buildx."""
    from .render_dockerfile import render_dockerfile

    typer.echo(f"→ Loading profile: {profile}")
    p = load_profile(profile)
    naming = load_naming()
    pkg, t = render_package_and_tag(p, naming)
    full_tag = f"{pkg}:{t}"
    profile_hash = compute_profile_hash(profile)
    typer.echo(f"→ Tag: {full_tag}")
    typer.echo(f"→ Hash: {profile_hash[:16]}...")

    rendered_path = render_dockerfile(p)
    typer.echo(f"→ Rendered Dockerfile: {rendered_path}")

    args = render_build_args(p, pkg, t, profile_hash)

    cmd: list[str] = ["docker", "buildx", "build", "-f", str(rendered_path)]
    cmd += ["-t", full_tag]
    for k, v in args.items():
        cmd += ["--build-arg", f"{k}={v}"]
    cmd += ["--push"] if push else ["--load"]
    cmd += [str(REPO_ROOT)]

    typer.echo(f"→ Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        typer.secho(f"✗ Build failed (exit {exc.returncode})", fg=typer.colors.RED)
        raise typer.Exit(code=exc.returncode) from exc

    typer.secho(f"✓ Build complete: {full_tag}", fg=typer.colors.GREEN)
    if not push:
        typer.echo(
            f"  Inspect: docker run --rm --entrypoint /bin/sh {full_tag} "
            "-c 'env | grep VLLM'"
        )


# --- `profile` subgroup ------------------------------------------------------


@profile_app.command("new")
def profile_new(
    gpu: str = typer.Option(..., help="GPU axis value (e.g., b300)"),
    model: str = typer.Option(..., help="Model family (e.g., kimi)"),
    model_revision: str = typer.Option(
        ..., help="Model revision (e.g., k26); must exist in tools/model-registry.cue"
    ),
    quant: str | None = typer.Option(None, help="Quantization (e.g., int4); omit for none"),
    mlnode: str = typer.Option(DEFAULT_MLNODE_VERSION, help="mlnode version"),
    vllm: str = typer.Option(DEFAULT_VLLM_VERSION, help="vLLM version"),
    rev: int = typer.Option(1, help="Kaitaku revision (k<N>)"),
) -> None:
    """Generate a new profile .cue file from template."""
    from .profile_factory import generate_profile

    path = generate_profile(
        gpu=gpu,
        model=model,
        model_revision=model_revision,
        quant=quant,
        mlnode=mlnode,
        vllm=vllm,
        rev=rev,
    )
    typer.secho(f"✓ Created profile: {path}", fg=typer.colors.GREEN)
    typer.echo(f"  Next: mlnode-foundry validate {path.stem}")


@profile_app.command("add-model")
def profile_add_model(
    model: str = typer.Argument(..., help="Model family (e.g., deepseek)"),
    model_revision: str = typer.Option(
        ..., help="Model revision (e.g., v3); applied to all generated profiles"
    ),
    gpus: str = typer.Option("b300,h100", help="Comma-separated GPUs to bulk-generate"),
    quant: str | None = typer.Option(None),
) -> None:
    """Bulk-generate profiles for `model:revision` across multiple GPUs."""
    from .profile_factory import bulk_add_model

    paths = bulk_add_model(
        model=model,
        model_revision=model_revision,
        gpus=gpus.split(","),
        quant=quant,
    )
    typer.secho(f"✓ Created {len(paths)} profile(s):", fg=typer.colors.GREEN)
    for p in paths:
        typer.echo(f"  {p}")


@profile_app.command("add-gpu")
def profile_add_gpu(
    gpu: str = typer.Argument(..., help="GPU (e.g., l40)"),
    models: str = typer.Option(
        "qwen:v3-235b,kimi:k26",
        help="Comma-separated <model>:<revision> pairs to bulk-generate",
    ),
) -> None:
    """Bulk-generate profiles for `gpu` across multiple (model, revision) pairs."""
    from .profile_factory import bulk_add_gpu

    pairs: list[tuple[str, str]] = []
    for pair in models.split(","):
        m, r = pair.split(":", 1)
        pairs.append((m, r))
    paths = bulk_add_gpu(gpu=gpu, models=pairs)
    typer.secho(f"✓ Created {len(paths)} profile(s):", fg=typer.colors.GREEN)
    for path in paths:
        typer.echo(f"  {path}")


# --- `runner` subgroup -------------------------------------------------------


@runner_app.command("list")
def runner_list() -> None:
    """List all registered runners from tools/runners.cue."""
    runners = list_runners()
    for name, runner in runners.items():
        kind = runner["kind"]
        budget = runner.get("budget", {})
        max_min = budget.get("max_runtime_min", "?")
        cost = budget.get("max_cost_per_run_usd")
        cost_str = f", ≤${cost}/run" if cost else ""
        typer.echo(f"  {name}  ({kind}, max {max_min} min{cost_str})")


@runner_app.command("select")
def runner_select(
    profile: str = typer.Argument(..., help="Profile name"),
    tier: str = typer.Option("smoke", help="Tier: smoke | benchmark"),
) -> None:
    """Pick the best runner for a profile + tier from registered runners."""
    name, runner = select_runner(profile, tier)
    typer.echo(f"Selected: {name}")
    typer.echo(json.dumps(runner, indent=2))


if __name__ == "__main__":
    app()
