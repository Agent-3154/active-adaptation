"""Robot bundles cooked from assetx recipes into ``ROBOT_MODEL_DIR``.

Factories call :func:`cooked_model_dir` when they build their config. It only
checks the bundle; cooking is always explicit via ``aa-cook-assets``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from active_adaptation import ROBOT_MODEL_DIR


class UncookedAssetError(FileNotFoundError):
    pass


def cooked_model_dir(recipe: str, *, usd: bool) -> Path:
    """Return ``ROBOT_MODEL_DIR/<recipe>`` if it holds a fresh bundle, else raise."""
    from assetx.cook import check

    out_dir = ROBOT_MODEL_DIR / recipe
    reason = check(recipe, out_dir, usd=usd)
    if reason is not None:
        raise UncookedAssetError(
            f"Robot bundle {out_dir} is {reason}. Cook it with:\n"
            f"    aa-cook-assets {recipe}"
        )
    return out_dir


def cook_assets(
    names: Annotated[
        Optional[list[str]],
        typer.Argument(help="Recipes to cook (default: every registered assetx recipe).", show_default=False),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if up to date.")] = False,
    no_usd: Annotated[bool, typer.Option("--no-usd", help="Skip the USD export (mjlab only).")] = False,
    check_only: Annotated[
        bool, typer.Option("--check", help="Only report status; exit 1 if any bundle needs cooking.")
    ] = False,
) -> None:
    """Cook assetx robot bundles into ROBOT_MODEL_DIR."""
    from assetx.cook import check, cook
    from assetx.recipes import list_recipes

    usd = not no_usd
    outdated = False
    for name in names or list_recipes():
        out_dir = ROBOT_MODEL_DIR / name
        reason = check(name, out_dir, usd=usd)
        if check_only or (reason is None and not force):
            typer.echo(f"{name}: {reason or 'up to date'} ({out_dir})")
            outdated |= reason is not None
            continue
        typer.echo(f"{name}: cooking ({reason or 'forced'}) -> {out_dir}")
        cook(name, out_dir, usd=usd, force=force)
    if outdated:
        raise typer.Exit(1)


def main() -> None:
    typer.run(cook_assets)
