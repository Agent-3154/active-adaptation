import active_adaptation


if active_adaptation.get_backend() == "isaac":
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import attach_stage_to_usd_context, use_stage

    def create_isaaclab_sim_and_scene(
        sim_cfg: SimulationCfg, scene_cfg: InteractiveSceneCfg
    ):
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            sim = SimulationContext(sim_cfg)
        else:
            raise RuntimeError(
                "Simulation context already exists. Cannot create a new one."
            )

        with use_stage(sim.get_initial_stage()):
            scene = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()

        sim.reset()
        sim.step(render=sim.has_gui())
        return sim, scene

elif active_adaptation.get_backend() == "mujoco":
    pass
else:
    raise NotImplementedError
