"""MJCF export helpers for the MotrixSim backend."""

from __future__ import annotations

from pathlib import Path

from mjlab.entity import Entity, EntityCfg


def export_entity_mjcf(cfg: EntityCfg, source_mjcf: Path | str) -> str:
    """Build an mjlab Entity and write an MJCF file next to the source asset.

    MotrixSim expects actuators in the MJCF, while active-adaptation defines them
    procedurally via mjlab ``EntityCfg``. This reuses mjlab's ``Entity`` pipeline
    (actuators, collision editors, keyframes) and exports the result to a sibling
    file so relative ``meshdir`` paths keep resolving.

    Example: ``a2/a2.xml`` -> ``a2/a2.motrix.mjcf.xml``
    """
    source_mjcf = Path(source_mjcf)
    entity = Entity(cfg)
    tmp_path = source_mjcf.with_name(f"{source_mjcf.stem}.motrix.mjcf.xml")
    entity.write_xml(tmp_path)
    return str(tmp_path)
