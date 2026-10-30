"""Cook AA-owned robot bundles into ``ROBOT_MODEL_DIR`` (``aa-cook-assets``).

Robots listed in :func:`list_recipes` are built by this CLI. Other bundles must
be installed under ``ROBOT_MODEL_DIR/<name>/`` separately;
:func:`cooked_model_dir` checks layout only for those names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from active_adaptation import ROBOT_MODEL_DIR
from assetx.cook import Recipe
from assetx.fetch import parse_fetch_method, parse_github_dir_url


class UncookedAssetError(FileNotFoundError):
    pass


def _require_pinned_vendor(url: str) -> str:
    ref = parse_github_dir_url(url)
    if len(ref.ref) != 40 or any(c not in "0123456789abcdef" for c in ref.ref):
        raise ValueError(f"Vendor URL must pin a full commit SHA, got {ref.ref!r}")
    return url


_RECIPES: dict[str, Recipe] | None = None


def _recipes() -> dict[str, Recipe]:
    global _RECIPES
    if _RECIPES is not None:
        return _RECIPES
    from active_adaptation.assets.quadrupeds.as2 import UNITREE_AS2_VENDOR, build_as2_mjcf
    from active_adaptation.assets.quadrupeds.spot import MENAGERIE_SPOT_VENDOR, build_spot_mjcf
    from active_adaptation.assets.quadrupeds.spot_arm import build_spot_arm_mjcf

    _RECIPES = {
        "as2": Recipe(
            name="as2",
            fn=build_as2_mjcf,
            vendor=_require_pinned_vendor(UNITREE_AS2_VENDOR),
            mesh_format="stl",
        ),
        "spot": Recipe(
            name="spot",
            fn=build_spot_mjcf,
            vendor=_require_pinned_vendor(MENAGERIE_SPOT_VENDOR),
        ),
        "spot_arm": Recipe(
            name="spot_arm",
            fn=build_spot_arm_mjcf,
            vendor=_require_pinned_vendor(MENAGERIE_SPOT_VENDOR),
        ),
    }
    return _RECIPES


def get_recipe(name: str) -> Recipe:
    try:
        return _recipes()[name]
    except KeyError:
        raise KeyError(f"Unknown aa-cook recipe {name!r}; available: {list_recipes()}") from None


def list_recipes() -> list[str]:
    return sorted(_recipes())


def _bundle_layout_check(out_dir: Path, *, usd: bool) -> str | None:
    if not out_dir.is_dir():
        return "missing"
    if not (out_dir / "model.xml").is_file():
        return "missing model.xml"
    if usd:
        usd_dir = out_dir / "usd"
        if not usd_dir.is_dir() or not any(usd_dir.glob("*.usdc")):
            return "missing USD"
    return None


def cooked_model_dir(name: str, *, usd: bool) -> Path:
    """Return ``ROBOT_MODEL_DIR/<name>`` if a usable bundle is present."""
    from assetx.cook import check

    out_dir = ROBOT_MODEL_DIR / name
    if name in _recipes():
        reason = check(name, out_dir, get_recipe=get_recipe, usd=usd)
        cook_hint = f"    aa-cook-assets {name}"
    else:
        reason = _bundle_layout_check(out_dir, usd=usd)
        cook_hint = f"    Install a bundle under {out_dir}/ (see assetx/examples/ if applicable)"
    if reason is not None:
        raise UncookedAssetError(f"Robot bundle {out_dir} is {reason}. Cook or install with:\n{cook_hint}")
    return out_dir


def cook_assets(
    names: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="Recipes to cook (default: every entry in RECIPES).",
            show_default=False,
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if up to date.")] = False,
    no_usd: Annotated[bool, typer.Option("--no-usd", help="Skip the USD export (mjlab only).")] = False,
    check_only: Annotated[
        bool, typer.Option("--check", help="Only report status; exit 1 if any bundle needs cooking.")
    ] = False,
    fetch: Annotated[
        Optional[str],
        typer.Option(
            "--fetch",
            help="Vendor download: auto, tarball, http, git (also ASSETX_FETCH). Default: auto.",
        ),
    ] = None,
) -> None:
    """Cook robots listed in ``RECIPES`` into ROBOT_MODEL_DIR."""
    from assetx.cook import check, cook

    if fetch is not None:
        os.environ["ASSETX_FETCH"] = parse_fetch_method(fetch)

    usd = not no_usd
    outdated = False
    for name in names or list_recipes():
        if name not in _recipes():
            raise typer.BadParameter(f"Unknown recipe {name!r}; aa-cook supports: {list_recipes()}")
        out_dir = ROBOT_MODEL_DIR / name
        reason = check(name, out_dir, get_recipe=get_recipe, usd=usd)
        if check_only or (reason is None and not force):
            typer.echo(f"{name}: {reason or 'up to date'} ({out_dir})")
            outdated |= reason is not None
            continue
        typer.echo(f"{name}: cooking ({reason or 'forced'}) -> {out_dir}")
        cook(name, out_dir, get_recipe=get_recipe, usd=usd, force=force)
    if outdated:
        raise typer.Exit(1)


def main() -> None:
    typer.run(cook_assets)
