from setuptools import find_namespace_packages, setup

packages = ["active_adaptation", "scripts"]
packages.extend(find_namespace_packages(include=["hydra_plugins.*"]))
packages.append("projects.facet")

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=packages,
    entry_points={
        "active_adaptation.projects": [
            "facet = projects.facet",
        ]
    },
    version="0.2.0",
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
