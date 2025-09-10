from setuptools import find_namespace_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=[
        "active_adaptation",
        "scripts",
    ] + find_namespace_packages(include=["hydra_plugins.*"]),
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
