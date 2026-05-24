"""NiceGUI rollout archive manager with staged key edits.

Launch::

    conda activate lab51
    python scripts/rollout_manager_nicegui.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from nicegui import ui

from active_adaptation.rollout_io import (
    DEFAULT_ROLLOUT_ROOT,
    apply_key_concat,
    apply_key_deletions,
    apply_key_renames,
    combine_rollouts,
    default_concat_key_name,
    delete_rollout_archive,
    discover_rollouts,
    format_bytes,
    get_tensor_entries,
    load_metadata,
    load_rollout,
    resolve_out_path,
    save_rollout_with_metadata,
    update_metadata_shapes,
    validate_combine,
    validate_delete_keys,
    validate_key_concat,
    validate_rename_map,
    key_from_str,
    list_tensor_keys,
)


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


@dataclass
class KeyRow:
    original_key: str
    name: str
    shape: str
    dtype: str
    size_human: str
    deleted: bool = False
    concat: bool = False


@dataclass
class ConcatOp:
    source_keys: list[str]
    dest_key: str
    dim: int = -1


def _save_stacked(path: Path, stacked, *, save_as: bool, suffix: str = "edited") -> Path:
    payload = load_rollout(path)
    payload = dict(payload)
    payload["stacked"] = stacked
    metadata = load_metadata(path.with_suffix(".json"))
    metadata = update_metadata_shapes(metadata, stacked)
    save_mode = "New file" if save_as else "In-place (no backup)"
    out_path = resolve_out_path(path, stacked, save_mode, suffix)
    save_rollout_with_metadata(out_path, payload, metadata)
    return out_path


def _common_key_entries(paths: list[str]) -> dict[str, dict]:
    shape_sets: list[set[str]] = []
    reference: dict[str, dict] = {}
    for pt_path in paths:
        path = Path(pt_path)
        payload = load_rollout(path)
        metadata = load_metadata(path.with_suffix(".json"))
        entries = get_tensor_entries(metadata, payload["stacked"])
        shape_sets.append(set(entries.keys()))
        if not reference:
            reference = entries
    common = set.intersection(*shape_sets) if shape_sets else set()
    return {key: reference[key] for key in sorted(common)}


class RolloutManagerUI:
    def __init__(self) -> None:
        self.selected: set[str] = set()
        self.editing_path: str | None = None
        self.key_rows: list[KeyRow] = []
        self.dirty = False
        self.suffix = "edited"
        self.concat_order: list[str] = []
        self.pending_concats: list[ConcatOp] = []
        self.concat_dim = -1
        self.status_label: ui.label | None = None
        self.combine_btn: ui.button | None = None
        self.save_btn: ui.button | None = None
        self.save_as_btn: ui.button | None = None
        self.revert_btn: ui.button | None = None
        self.dirty_label: ui.label | None = None
        self.keys_info_label: ui.label | None = None
        self.concat_dest_input: ui.input | None = None
        self.concat_btn: ui.button | None = None

    def set_status(self, msg: str, *, kind: str = "info") -> None:
        if self.status_label is not None:
            self.status_label.set_text(msg)
        ui.notify(msg.split("\n")[0], type=kind)

    def _update_action_buttons(self) -> None:
        if self.combine_btn is not None:
            self.combine_btn.enable() if len(self.selected) >= 2 else self.combine_btn.disable()

        single = len(self.selected) == 1 and self.editing_path is not None
        has_edits = self.dirty and single
        if self.save_btn is not None:
            self.save_btn.enable() if has_edits else self.save_btn.disable()
        if self.save_as_btn is not None:
            self.save_as_btn.enable() if has_edits else self.save_as_btn.disable()
        if self.revert_btn is not None:
            self.revert_btn.enable() if has_edits else self.revert_btn.disable()
        if self.dirty_label is not None:
            self.dirty_label.set_text("Unsaved changes" if has_edits else "")

        if self.save_as_btn is not None and single:
            path = Path(self.editing_path)
            payload = load_rollout(path)
            name = self._save_as_name(path, payload["stacked"])
            self.save_as_btn.set_text(f"Save As ({name})")

        if self.concat_btn is not None:
            self.concat_btn.enable() if single and len(self.concat_order) >= 2 else self.concat_btn.disable()

    def _reset_concat_state(self) -> None:
        self.concat_order = []
        self.pending_concats = []
        if self.concat_dest_input is not None:
            self.concat_dest_input.set_value("")

    def _display_name(self, row: KeyRow) -> str:
        return row.name.strip() or row.original_key

    def _default_concat_dest(self) -> str:
        names: list[str] = []
        for key in self.concat_order:
            row = next((r for r in self.key_rows if r.original_key == key), None)
            if row is not None and not row.deleted:
                names.append(self._display_name(row))
        return default_concat_key_name(names)

    def _update_concat_dest_default(self) -> None:
        if self.concat_dest_input is None:
            return
        if self.concat_order:
            self.concat_dest_input.set_value(self._default_concat_dest())
        else:
            self.concat_dest_input.set_value("")

    def _stacked_with_pending_concats(self, stacked):
        out = stacked
        for op in self.pending_concats:
            out = apply_key_concat(out, op.source_keys, op.dest_key, dim=op.dim)
        return out

    def _entry_from_tensor(self, tensor) -> tuple[str, str, str]:
        shape = str(list(tensor.shape))
        dtype = str(tensor.dtype).removeprefix("torch.")
        size_human = format_bytes(int(tensor.numel() * tensor.element_size()))
        return shape, dtype, size_human

    def _load_keys(self, path: str) -> None:
        pt_path = Path(path)
        payload = load_rollout(pt_path)
        stacked = payload["stacked"]
        metadata = load_metadata(pt_path.with_suffix(".json"))
        entries = get_tensor_entries(metadata, stacked)
        self.editing_path = _path_key(path)
        self.key_rows = [
            KeyRow(
                original_key=key,
                name=key,
                shape=str(info["shape"]),
                dtype=str(info["dtype"]),
                size_human=format_bytes(int(info["size_bytes"])),
            )
            for key, info in sorted(entries.items())
        ]
        self._reset_concat_state()
        self.dirty = False
        self._update_keys_info()
        self._update_action_buttons()

    def _update_keys_info(self) -> None:
        if self.keys_info_label is None or self.editing_path is None:
            return
        path = Path(self.editing_path)
        payload = load_rollout(path)
        stacked = payload["stacked"]
        self.keys_info_label.set_text(
            f"{path.name}: T={stacked.batch_size[0]}, N={stacked.batch_size[1]}, "
            f"{len(self.key_rows)} keys, {format_bytes(path.stat().st_size)}"
        )

    def _ops_from_rows(self) -> tuple[list[str], dict[str, str]]:
        deletes = [row.original_key for row in self.key_rows if row.deleted]
        renames: dict[str, str] = {}
        for row in self.key_rows:
            if row.deleted:
                continue
            new_name = row.name.strip()
            if new_name and new_name != row.original_key:
                renames[row.original_key] = new_name
        return deletes, renames

    def _validate_rows(self, stacked) -> list[str]:
        deletes, renames = self._ops_from_rows()
        errors: list[str] = []
        if not deletes and not renames and not self.pending_concats:
            errors.append("No edits to save.")
            return errors

        trial = self._stacked_with_pending_concats(stacked)

        overlap = set(deletes) & set(renames.keys())
        if overlap:
            errors.append(f"Keys marked deleted and renamed: {sorted(overlap)}")

        if deletes:
            _, delete_errors = validate_delete_keys(trial, deletes)
            errors.extend(delete_errors)

        if renames:
            remaining = trial
            if deletes:
                try:
                    remaining = apply_key_deletions(trial, deletes)
                except ValueError:
                    pass
            _, rename_errors = validate_rename_map(remaining, renames)
            errors.extend(rename_errors)

        return errors

    def _apply_rows(self, stacked):
        out = self._stacked_with_pending_concats(stacked)
        deletes, renames = self._ops_from_rows()
        if deletes:
            out = apply_key_deletions(out, deletes)
        if renames:
            out = apply_key_renames(out, renames)
        return out

    def _save_as_name(self, path: Path, stacked) -> str:
        T, N = int(stacked.batch_size[0]), int(stacked.batch_size[1])
        tag = self.suffix.strip() or "edited"
        return f"rollout_{T}_{N}_{tag}.pt"

    async def _confirm(self, title: str, message: str) -> bool:
        with ui.dialog() as dialog, ui.card():
            ui.label(title).classes("text-lg font-medium")
            ui.label(message).classes("whitespace-pre-wrap")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False))
                ui.button("Confirm", color="negative", on_click=lambda: dialog.submit(True))
        return bool(await dialog)

    async def _maybe_discard(self) -> bool:
        if not self.dirty:
            return True
        return await self._confirm(
            "Discard unsaved changes?",
            "Your staged key edits will be lost.",
        )

    async def _toggle_selected(self, path: str, selected: bool) -> None:
        key = _path_key(path)
        if selected:
            if self.dirty and self.editing_path != key:
                if not await self._maybe_discard():
                    self.rollout_list.refresh()
                    return
            self.selected.add(key)
        else:
            if self.dirty and self.editing_path == key:
                if not await self._maybe_discard():
                    self.rollout_list.refresh()
                    return
            self.selected.discard(key)
            if self.editing_path == key:
                self.editing_path = None
                self.key_rows = []
                self.dirty = False

        if len(self.selected) == 1:
            only = next(iter(self.selected))
            if self.editing_path != only:
                self._load_keys(only)
        elif len(self.selected) != 1:
            self.editing_path = None
            self.key_rows = []
            self.dirty = False
            if self.keys_info_label is not None:
                self.keys_info_label.set_text("")

        self._update_action_buttons()
        self.keys_panel.refresh()

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_action_buttons()

    def _toggle_key_concat(self, index: int, selected: bool) -> None:
        row = self.key_rows[index]
        if row.deleted:
            return
        row.concat = selected
        key = row.original_key
        if selected:
            if key not in self.concat_order:
                self.concat_order.append(key)
        else:
            self.concat_order = [k for k in self.concat_order if k != key]
        self._update_concat_dest_default()
        self._update_action_buttons()
        self.keys_panel.refresh()

    async def stage_concat(self) -> None:
        if self.editing_path is None or len(self.concat_order) < 2:
            self.set_status("Select at least two keys to concat.", kind="warning")
            return

        path = Path(self.editing_path)
        stacked = load_rollout(path)["stacked"]
        trial = self._stacked_with_pending_concats(stacked)
        dest_key = (
            self.concat_dest_input.value.strip()
            if self.concat_dest_input is not None
            else ""
        ) or self._default_concat_dest()
        source_keys = list(self.concat_order)

        existing = set(list_tensor_keys(trial))
        will_overwrite = dest_key in existing and dest_key not in source_keys
        if will_overwrite:
            if not await self._confirm(
                "Overwrite existing key?",
                f"Key {dest_key!r} already exists.\n\n"
                "Confirm to overwrite with the concat result, or cancel.",
            ):
                return

        _, dest_key, errors = validate_key_concat(
            trial,
            source_keys,
            dest_key,
            dim=self.concat_dim,
            allow_overwrite=will_overwrite,
        )
        if errors:
            self.set_status("\n".join(errors), kind="negative")
            return

        merged = apply_key_concat(
            trial, source_keys, dest_key, dim=self.concat_dim, allow_overwrite=will_overwrite
        )
        merged_tensor = merged.get(key_from_str(dest_key))
        shape, dtype, size_human = self._entry_from_tensor(merged_tensor)

        self.pending_concats = [
            op for op in self.pending_concats if op.dest_key != dest_key
        ]
        self.pending_concats.append(
            ConcatOp(source_keys=source_keys, dest_key=dest_key, dim=self.concat_dim)
        )
        for row in self.key_rows:
            if row.original_key in source_keys:
                row.concat = False

        existing_row = next(
            (row for row in self.key_rows if row.original_key == dest_key),
            None,
        )
        if existing_row is not None:
            existing_row.name = dest_key
            existing_row.shape = shape
            existing_row.dtype = dtype
            existing_row.size_human = size_human
            existing_row.deleted = False
        else:
            self.key_rows.append(
                KeyRow(
                    original_key=dest_key,
                    name=dest_key,
                    shape=shape,
                    dtype=dtype,
                    size_human=size_human,
                )
            )
            self.key_rows.sort(key=lambda row: row.original_key)

        self.concat_order = []
        self._update_concat_dest_default()
        self._mark_dirty()
        self.keys_panel.refresh()
        action = "Overwrote" if will_overwrite else "Staged concat ->"
        self.set_status(
            f"{action} {dest_key!r} from {source_keys}",
            kind="positive",
        )

    def _toggle_key_deleted(self, index: int) -> None:
        row = self.key_rows[index]
        if row.original_key in self.concat_order:
            self.concat_order = [k for k in self.concat_order if k != row.original_key]
            row.concat = False
            self._update_concat_dest_default()
        row.deleted = not row.deleted
        self._mark_dirty()
        self.keys_panel.refresh()

    def _on_key_name_change(self, index: int, value: str) -> None:
        if self.key_rows[index].name != value:
            self.key_rows[index].name = value
            if self.key_rows[index].original_key in self.concat_order:
                self._update_concat_dest_default()
            self._mark_dirty()

    def revert_edits(self) -> None:
        if self.editing_path is None:
            return
        self._load_keys(self.editing_path)
        self.keys_panel.refresh()
        self.set_status("Reverted changes.", kind="info")

    def save(self) -> None:
        if self.editing_path is None or not self.dirty:
            return
        path = Path(self.editing_path)
        payload = load_rollout(path)
        stacked = payload["stacked"]
        errors = self._validate_rows(stacked)
        if errors:
            self.set_status("\n".join(errors), kind="negative")
            return
        edited = self._apply_rows(stacked)
        out_path = _save_stacked(path, edited, save_as=False)
        self._reselect([str(out_path.resolve())])
        self.set_status(f"Saved {out_path.name}", kind="positive")

    def save_as(self) -> None:
        if self.editing_path is None or not self.dirty:
            return
        path = Path(self.editing_path)
        payload = load_rollout(path)
        stacked = payload["stacked"]
        errors = self._validate_rows(stacked)
        if errors:
            self.set_status("\n".join(errors), kind="negative")
            return
        edited = self._apply_rows(stacked)
        out_path = _save_stacked(path, edited, save_as=True, suffix=self.suffix)
        self._reselect([str(out_path.resolve())])
        self.set_status(f"Saved as {out_path.name}", kind="positive")

    def _reselect(self, paths: list[str]) -> None:
        self.selected = {_path_key(p) for p in paths}
        self.editing_path = next(iter(self.selected)) if len(self.selected) == 1 else None
        if self.editing_path is not None:
            self._load_keys(self.editing_path)
        else:
            self.key_rows = []
            self.dirty = False
            if self.keys_info_label is not None:
                self.keys_info_label.set_text("")
        self._update_action_buttons()
        self.rollout_list.refresh()
        self.keys_panel.refresh()

    async def delete_rollout(self, path: str) -> None:
        pt_path = Path(path)
        if not await self._confirm(
            "Delete rollout archive?",
            f"This permanently removes:\n  {pt_path}\n  {pt_path.with_suffix('.json')}",
        ):
            return
        delete_rollout_archive(pt_path)
        if self.editing_path == _path_key(path):
            self.editing_path = None
            self.key_rows = []
            self.dirty = False
            if self.keys_info_label is not None:
                self.keys_info_label.set_text("")
        self.selected.discard(_path_key(path))
        self.set_status(f"Deleted {pt_path.name}", kind="positive")
        self.rollout_list.refresh()
        self.keys_panel.refresh()
        self._update_action_buttons()

    async def combine_selected(self) -> None:
        if self.dirty and not await self._maybe_discard():
            return
        paths = sorted(self.selected)
        if len(paths) < 2:
            self.set_status("Select at least two rollouts.", kind="warning")
            return
        path_objs = [Path(p) for p in paths]
        _, errors = validate_combine(path_objs)
        if errors:
            self.set_status("Cannot combine:\n- " + "\n- ".join(errors), kind="negative")
            return
        if not await self._confirm(
            "Combine rollouts?",
            f"Merge {len(paths)} rollouts into one archive?",
        ):
            return
        try:
            out_path, metadata = combine_rollouts(path_objs)
        except ValueError as exc:
            self.set_status(str(exc), kind="negative")
            return
        self._reselect([str(out_path)])
        self.set_status(
            f"Merged -> {out_path.name} (policy={metadata.get('policy_name')})",
            kind="positive",
        )

    @ui.refreshable
    def rollout_list(self) -> None:
        rollouts = discover_rollouts()
        if not rollouts:
            ui.label("No rollout archives found.").classes("text-gray-500")
            return

        header = ui.row().classes("w-full gap-2 text-sm font-semibold text-gray-600 px-1")
        with header:
            ui.label("").style("width: 36px")
            ui.label("Task / run").classes("flex-1")
            ui.label("T×N").style("width: 72px")
            ui.label("Policy").classes("flex-1")
            ui.label("Size").style("width: 72px")
            ui.label("Keys").style("width: 48px")
            ui.label("").style("width: 48px")

        for rollout in rollouts:
            path = str(rollout.pt_path)
            path_key = _path_key(path)
            row = ui.row().classes(
                "w-full gap-2 items-center py-1 px-1 rounded "
                + ("bg-blue-50" if path_key in self.selected else "")
            )
            with row:
                ui.checkbox(
                    value=path_key in self.selected,
                    on_change=lambda e, p=path: self._toggle_selected(p, e.value),
                ).props("dense")
                with ui.column().classes("flex-1 gap-0"):
                    ui.label(f"{rollout.task_algo} / {rollout.timestamp}").classes("text-sm")
                    ui.label(rollout.pt_path.name).classes("text-xs text-gray-500")
                ui.label(f"{rollout.num_steps}×{rollout.num_envs}").style("width: 72px")
                ui.label(rollout.policy_name or "—").classes("flex-1 text-sm truncate")
                ui.label(rollout.size_human).style("width: 72px")
                ui.label(str(rollout.num_keys)).style("width: 48px")
                ui.button(
                    icon="delete",
                    color="negative",
                    on_click=lambda p=path: self.delete_rollout(p),
                ).props("flat dense round").tooltip("Delete archive")

    @ui.refreshable
    def keys_panel(self) -> None:
        selected = sorted(self.selected)
        n = len(selected)

        if n == 0:
            ui.label("Select a rollout to inspect keys.").classes("text-gray-500")
            return

        if n == 1:
            path_key = _path_key(selected[0])
            if self.editing_path != path_key:
                self._load_keys(selected[0])

            header = ui.row().classes("w-full gap-2 text-sm font-semibold text-gray-600")
            with header:
                ui.label("").style("width: 36px")
                ui.label("Key").classes("flex-1")
                ui.label("Shape").classes("flex-1")
                ui.label("Dtype").style("width: 88px")
                ui.label("Size").style("width: 72px")
                ui.label("").style("width: 48px")

            for index, row in enumerate(self.key_rows):
                row_classes = "w-full gap-2 items-center py-1 px-1 rounded"
                if row.deleted:
                    row_classes += " bg-red-50"
                elif row.original_key in {op.dest_key for op in self.pending_concats}:
                    row_classes += " bg-green-50"
                elif row.name.strip() != row.original_key:
                    row_classes += " bg-amber-50"

                with ui.row().classes(row_classes):
                    if row.deleted:
                        ui.label("").style("width: 36px")
                        ui.label(row.name).classes(
                            "flex-1 text-sm font-mono line-through text-red-700"
                        )
                    else:
                        order = (
                            str(self.concat_order.index(row.original_key) + 1)
                            if row.original_key in self.concat_order
                            else ""
                        )
                        with ui.row().classes("items-center").style("width: 36px"):
                            ui.checkbox(
                                value=row.concat,
                                on_change=lambda e, i=index: self._toggle_key_concat(i, e.value),
                            ).props("dense")
                            if order:
                                ui.label(order).classes("text-xs text-blue-600")
                        ui.input(
                            value=row.name,
                            on_change=lambda e, i=index: self._on_key_name_change(i, e.value),
                        ).classes("flex-1").props("dense")
                    ui.label(row.shape).classes("flex-1 text-sm text-gray-600 font-mono")
                    ui.label(row.dtype).classes("text-sm text-gray-600").style("width: 88px")
                    ui.label(row.size_human).classes("text-sm text-gray-600").style("width: 72px")
                    icon = "undo" if row.deleted else "delete"
                    tooltip = "Restore key" if row.deleted else "Mark for deletion"
                    ui.button(
                        icon=icon,
                        color="negative" if not row.deleted else "primary",
                        on_click=lambda i=index: self._toggle_key_deleted(i),
                    ).props("flat dense round").tooltip(tooltip)
            return

        common = _common_key_entries(selected)
        ui.label(f"{n} rollouts selected — {len(common)} common key(s).").classes("text-sm text-gray-700 mb-2")
        if not common:
            ui.label("No common keys.").classes("text-gray-500")
            return
        header = ui.row().classes("w-full gap-2 text-sm font-semibold text-gray-600")
        with header:
            ui.label("Key").classes("flex-1")
            ui.label("Shape").classes("flex-1")
            ui.label("Dtype").style("width: 88px")
            ui.label("Size").style("width: 72px")
        for key, info in common.items():
            with ui.row().classes("w-full gap-2"):
                ui.label(key).classes("flex-1 text-sm font-mono")
                ui.label(str(info["shape"])).classes("flex-1 text-sm text-gray-600 font-mono")
                ui.label(str(info["dtype"])).classes("text-sm text-gray-600").style("width: 88px")
                ui.label(format_bytes(int(info["size_bytes"]))).classes("text-sm text-gray-600").style("width: 72px")

    def build(self) -> None:
        ui.page_title("Rollout Manager")
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
            ui.label("Rollout Manager").classes("text-2xl font-bold")
            ui.label(f"Archives root: {DEFAULT_ROLLOUT_ROOT}").classes("text-sm text-gray-600")

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Refresh", icon="refresh", on_click=self.rollout_list.refresh).props("outline")
                self.combine_btn = ui.button(
                    "Combine selected",
                    icon="call_merge",
                    on_click=self.combine_selected,
                ).props("color=primary")
                self.combine_btn.disable()
                self.status_label = ui.label("").classes("flex-1 text-sm text-gray-700")

            ui.separator()
            ui.label("Rollouts").classes("text-lg font-medium")
            self.rollout_list()

            ui.separator()
            ui.label("Keys").classes("text-lg font-medium")
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                self.keys_info_label = ui.label("").classes("text-sm text-gray-700 flex-1")
                self.dirty_label = ui.label("").classes("text-sm text-orange-600")
                self.revert_btn = ui.button("Revert", on_click=self.revert_edits).props("flat dense")
                self.save_btn = ui.button("Save", on_click=self.save).props("color=primary dense")
                self.save_as_btn = ui.button("Save As", on_click=self.save_as).props("outline dense")
                self.revert_btn.disable()
                self.save_btn.disable()
                self.save_as_btn.disable()
            with ui.row().classes("w-full items-end gap-2 mb-2"):
                self.concat_dest_input = ui.input(
                    label="Concat as",
                    placeholder="Select keys in order, name auto-fills",
                ).classes("flex-1")
                ui.number(
                    label="Dim",
                    value=self.concat_dim,
                    on_change=lambda e: setattr(self, "concat_dim", int(e.value)),
                ).classes("w-24").props("dense")
                self.concat_btn = ui.button("Concat selected", on_click=self.stage_concat).props("outline dense")
                self.concat_btn.disable()
            self.keys_panel()


@ui.page("/")
def index() -> None:
    RolloutManagerUI().build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout archive manager (NiceGUI)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    ui.run(title="Rollout Manager", host=args.host, port=args.port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
