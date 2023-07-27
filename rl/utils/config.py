import collections
from copy import deepcopy
import yaml
import os
from typing import Any


def config_copy(config: Any):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def read_yaml(dirpath, filename):

    filepath = os.path.join(dirpath, filename)
    if not os.path.isfile(filepath): return None

    with open(filepath, 'r') as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(filename, exc)
    return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_config(agent_name: str, env_name: str):

    # {agent}/{env}.yaml
    filename = f'{env_name}.yaml'
    dirpath = os.path.join('rl', 'config', 'agents', f'{agent_name}')

    config_dict = read_yaml(dirpath, filename)
    return config_dict


def save_config(config):
    # make directory for config saving
    config_dirpath = os.path.join(
        os.getcwd(),
        config.local_results_path,
        "models",
        config.unique_token
    )
    os.makedirs(config_dirpath, exist_ok=True)

    config_filepath = os.path.join(
        config_dirpath,
        "{}.yaml".format(config.unique_token))

    if os.path.isfile(config_filepath) : return

    with open(config_filepath, 'w', encoding="utf-8") as file:
        for key, value in config.__dict__.items():
            if isinstance(value, str):
                file.write(f"{key}: '{value}' \n")
            else:
                file.write(f"{key}: {value} \n")
        file.close()