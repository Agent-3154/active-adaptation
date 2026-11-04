"""Fixed mesh variants using mjlab's mesh merger and per-world realization.

The upstream floating-base restriction prevents fixed entities from stacking at
the origin. These entities instead have an explicit mocap root and no runtime
DOFs. The temporary free joint exists only while merging mesh slots, never in
the compiled scene or the source specs used to compute physical constants.
"""
from dataclasses import dataclass, replace

import mujoco
from mjlab.entity import Entity
from mjlab.entity.variants import VariantEntityCfg, build_merged_variant_spec


@dataclass
class FixedVariantEntityCfg(VariantEntityCfg):
    def build(self):
        return FixedVariantEntity(self)


class FixedVariantEntity(Entity):
    def _build_spec(self):
        def for_merge(factory):
            def build():
                spec = factory().copy()
                roots = list(spec.worldbody.bodies)
                if len(roots) != 1 or list(spec.joints):
                    raise ValueError('Fixed variants require one root and no joints')
                if list(spec.keys):
                    raise ValueError('Fixed variant source specs cannot contain keyframes')
                root = roots[0]
                root.mocap = False
                root.add_freejoint(name='aa_fixed_variant_merge_joint')
                return spec
            return build

        merge_cfg = VariantEntityCfg(
            variants={name: for_merge(fn) for name, fn in self.cfg.variants.items()},
            assignment=self.cfg.assignment,
        )
        self._spec, metadata = build_merged_variant_spec(merge_cfg)

        def fix(spec):
            root = list(spec.worldbody.bodies)[0]
            joints = list(root.joints)
            if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('Unexpected fixed-variant merged joint structure')
            spec.delete(joints[0])
            root.mocap = True
            # Fixed root initialization is not included in a qpos keyframe.
            root.pos = self.cfg.init_state.pos
            root.quat = self.cfg.init_state.rot
            return spec

        fix(self._spec)
        self._variant_metadata = replace(
            metadata, variant_source_specs=tuple(fix(s.copy()) for s in metadata.variant_source_specs)
        )
