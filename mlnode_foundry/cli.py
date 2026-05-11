"""mlnode-foundry CLI — Typer entrypoint."""

from __future__ import annotations

import json
import subprocess

import typer

from .build_hash import compute_profile_hash
from .expand import expand_profile
from .render_bake import REPO_ROOT, load_profile, render_build_args
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
    """Print self-contained resolved profile (all bases inlined) as JSON."""
    resolved = expand_profile(profile)
    typer.echo(json.dumps(resolved, indent=2))


@app.command()
def build(
    profile: str,
    push: bool = typer.Option(
        False, "--push/--no-push", help="Push to registry after build (default: no push)."
    ),
) -> None:
    """Build a profile via docker buildx."""
    typer.echo(f"→ Loading profile: {profile}")
    p = load_profile(profile)
    naming = load_naming()
    pkg, t = render_package_and_tag(p, naming)
    full_tag = f"{pkg}:{t}"
    profile_hash = compute_profile_hash(profile)
    typer.echo(f"→ Tag: {full_tag}")
    typer.echo(f"→ Hash: {profile_hash[:16]}...")

    args = render_build_args(p, pkg, t, profile_hash)

    cmd: list[str] = ["docker", "buildx", "build", "-f", str(REPO_ROOT / "stage3" / "Dockerfile")]
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
        typer.echo(f"  Inspect: docker run --rm --entrypoint /bin/sh {full_tag} -c 'env | grep VLLM'")


# --- `profile` subgroup ------------------------------------------------------


@profile_app.command("new")
def profile_new(
    gpu: str = typer.Option(..., help="GPU axis value (e.g., b300)"),
    model: str = typer.Option(..., help="Model axis value (e.g., kimi)"),
    quant: str | None = typer.Option(None, help="Quantization (e.g., int4); omit for none"),
    mlnode: str = typer.Option("0.2.13", help="mlnode version"),
    vllm: str = typer.Option("0.20.0", help="vLLM version"),
    rev: int = typer.Option(1, help="Kaitaku revision (k<N>)"),
) -> None:
    """Generate a new profile .cue file from template."""
    from .profile_factory import generate_profile

    path = generate_profile(gpu=gpu, model=model, quant=quant, mlnode=mlnode, vllm=vllm, rev=rev)
    typer.secho(f"✓ Created profile: {path}", fg=typer.colors.GREEN)
    typer.echo(f"  Next: mlnode-foundry validate {path.stem}")


@profile_app.command("add-model")
def profile_add_model(
    model: str = typer.Argument(..., help="Model family (e.g., deepseek)"),
    gpus: str = typer.Option("b300,h100", help="Comma-separated GPUs to bulk-generate"),
    quant: str | None = typer.Option(None),
) -> None:
    """Bulk-generate profiles for `model` across multiple GPUs."""
    from .profile_factory import bulk_add_model

    paths = bulk_add_model(model=model, gpus=gpus.split(","), quant=quant)
    typer.secho(f"✓ Created {len(paths)} profile(s):", fg=typer.colors.GREEN)
    for p in paths:
        typer.echo(f"  {p}")


@profile_app.command("add-gpu")
def profile_add_gpu(
    gpu: str = typer.Argument(..., help="GPU (e.g., l40)"),
    models: str = typer.Option("qwen,kimi", help="Comma-separated models to bulk-generate"),
) -> None:
    """Bulk-generate profiles for `gpu` across multiple models."""
    from .profile_factory import bulk_add_gpu

    paths = bulk_add_gpu(gpu=gpu, models=models.split(","))
    typer.secho(f"✓ Created {len(paths)} profile(s):", fg=typer.colors.GREEN)
    for p in paths:
        typer.echo(f"  {p}")


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
