"""mlnode-foundry CLI — Typer entrypoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from .render_bake import (
    REPO_ROOT,
    load_naming,
    load_profile,
    render_build_args,
    render_package_and_tag,
)
from .validate import validate_profile

app = typer.Typer(
    help="mlnode-foundry: build pipeline for Kaitaku-tuned mlnode container images.",
    no_args_is_help=True,
)


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
def build(
    profile: str,
    push: bool = typer.Option(
        False, "--push/--no-push", help="Push to registry after build (default: no push)."
    ),
) -> None:
    """Build a profile via docker buildx (Discovery spike: skips Stage 2, uses upstream as base)."""
    typer.echo(f"→ Loading profile: {profile}")
    p = load_profile(profile)
    naming = load_naming()
    pkg, t = render_package_and_tag(p, naming)
    full_tag = f"{pkg}:{t}"
    typer.echo(f"→ Tag: {full_tag}")

    args = render_build_args(p, pkg, t)

    cmd: list[str] = ["docker", "buildx", "build"]
    cmd += ["-f", str(REPO_ROOT / "stage3" / "Dockerfile")]
    cmd += ["-t", full_tag]
    for k, v in args.items():
        cmd += ["--build-arg", f"{k}={v}"]
    if push:
        cmd += ["--push"]
    else:
        cmd += ["--load"]
    cmd += [str(REPO_ROOT)]

    typer.echo(f"→ Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        typer.secho(f"✗ Build failed (exit {exc.returncode})", fg=typer.colors.RED)
        raise typer.Exit(code=exc.returncode)

    typer.secho(f"✓ Build complete: {full_tag}", fg=typer.colors.GREEN)
    if not push:
        typer.echo(f"  Run: docker run --rm {full_tag} sh -c 'env | grep VLLM'")


if __name__ == "__main__":
    app()
