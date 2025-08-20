from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    package_dir={
        "active_adaptation": "active_adaptation",
        "active_adaptation_projects": "projects",
    },
    packages=[
        "active_adaptation",
        "active_adaptation_projects",
        "scripts",
    ],
    version="0.1.3",
    install_requires=[
        "setproctitle",
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "av", # for moviepy
        "einops",
        "termcolor",
        "pygame", # for game controller
        "tensordict",
        "torchrl",
        "mujoco",
        "linuxfd",
    ],
)
