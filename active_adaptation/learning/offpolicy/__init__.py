import os
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if file.endswith(".py") and not file.startswith("_"):
        importlib.import_module(f".{file[:-3]}", __package__)
