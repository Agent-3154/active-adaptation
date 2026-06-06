import active_adaptation
from omegaconf import OmegaConf

def main():
    active_adaptation.init(
        cfg=OmegaConf.create({
            "backend": "motrix",
        }),
        auto_rank=False,
    )
    from active_adaptation.registry import Registry
    registry = Registry.instance()
    cfg, _sensors = registry.get("asset", "unitree_a2")(backend="motrix")

    from active_adaptation.envs.backends.motrix import (
        MotrixSceneCfg,
        MotrixScene,
    )
    from motrixsim import run
    from motrixsim.render import RenderApp

    scene_cfg = MotrixSceneCfg(
        num_envs=4,
        env_spacing=2.0,
        entities={"robot": cfg},
    )
    scene = MotrixScene(scene_cfg)
    print(f"MSD model: {scene.msd_model}")
    print(scene.msd_model.num_actuators)

    with RenderApp() as render:
        render.launch(
            scene.msd_model,
            batch=scene.cfg.num_envs,
            render_offset=scene.render_offsets,
        )
        def step():
            scene.msd_model.step(scene.msd_data)
            scene.update(scene.msd_model.options.timestep)
            for ent in scene.entities.values():
                print(ent.data.root_link_pos_w)

        run.render_loop(
            scene.msd_model.options.timestep,
            60.0,
            step,
            lambda: render.sync(scene.msd_data),
        )

if __name__ == "__main__":
    main()