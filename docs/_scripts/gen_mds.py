"""
   isort:skip_file
"""

import os

from tqdm import tqdm

import gymnasium as gym

from importlib import import_module
from utils import trim
import re


# REWRITE: generate md's for new environments that don't belong to Fetch or Shadow Hand
# TODO: use same format for Fetch and Shadow Hand
# The environment entrypoints have the following standard: `gymnasium_robotics.envs.env_type.env_name:EnvName`
all_envs = []
for env_spec in gym.envs.registry.values():
    if isinstance(env_spec.entry_point, str):
        if (
            env_spec.entry_point.startswith("gymnasium_robotics.envs")
            and "MujocoPy" not in env_spec.entry_point
        ):
            all_envs.append(env_spec)  # Exclude Fetch and Shadow Hand environments

# Keep latest version of environments
filtered_envs_by_name = {}
for env_spec in all_envs:
    if env_spec.name not in filtered_envs_by_name:
        filtered_envs_by_name[env_spec.name] = env_spec
    elif filtered_envs_by_name[env_spec.name].version < env_spec.version:
        filtered_envs_by_name[env_spec.name] = env_spec

# Extract non-repeated entrypoints of environments
entry_points = {env_spec.entry_point for env_spec in filtered_envs_by_name.values()}

for entry_point in tqdm(entry_points):
    split_module_class = entry_point.split(":")
    module = split_module_class[0]
    env_class = split_module_class[1]
    docstring = getattr(import_module(module), env_class).__doc__
    docstring = trim(docstring)

    split_entrypoint = module.split(".")
    if len(split_entrypoint) == 4:
        env_type = split_entrypoint[-2]
        env_name = split_entrypoint[-1]

    if len(split_entrypoint) == 3:
        env_type = split_entrypoint[-1]
        env_name = split_entrypoint[-1]
    
    # Remove file version from env_name
    env_name = re.sub('_v(?P<version>\d+)', '', env_name)
    title_env_name = env_name.replace("_", " ").title()

    v_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "envs",
        env_type,
        env_name + ".md",
    )

    front_matter = f"""---
autogenerated:
title: {title_env_name}
---
"""
    title = f"# {title_env_name}"
    gif = (
        "```{figure}"
        + f" ../../_static/videos/{env_type}/{env_name}.gif"
        + f" \n:width: 200px\n:name: {env_name}\n```"
    )
    all_text = f"""{front_matter}
{title}

{gif}

{docstring}
"""

    file = open(v_path, "w", encoding="utf-8")
    file.write(all_text)
    file.close()
