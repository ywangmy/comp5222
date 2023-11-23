#
# Created on Thu Jun 29 2023 18:56:15
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#
import os

import cv2
import numpy as np
import torch


loaded_models = []  # list of loaded models for one total program run


def load_model(model_path: str) -> torch.nn.Module:
    """ensure that only one instance of the model is loaded, later loadings will point to the same model instance loaded before

    Args:
        model_path (str): path to the model

    Returns:
        torch.nn.Module: model
    """
    model = torch.jit.load(model_path)

    for loaded_model in loaded_models:
        if (
            model.state_dict() == loaded_model.state_dict()
        ):  # if the model is already loaded
            del model  # delete the duplicate model
            return loaded_model  # return the loaded model

    loaded_models.append(model)  # model's a new model, add it to the list
    return model


def print_dict(d: dict, indent: int = 0) -> None:
    """print a dictionary with indentation, uses recursion to print nested dictionaries

    Args:
        d (dict): dictionary to be printed
        indent (int, optional): indentation level. Defaults to 0.

    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + str(key) + ": ")
            print_dict(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(
                "  " * indent + str(key) + ":\n" + str(value)
            )  # new line before printing matrix for better readability
        else:
            print("  " * indent + str(key) + ": " + str(value))


def parse_path(probe_path: str, base_path: str = None):
    """parse a potential path, expand ~ to user home, and check if the path exists

    Args:
        probe_path (str): the path to be parsed
        base_path (str, optional): the base path of the current (yaml) file. Defaults to None.

    Returns:
        the parsed path if it exists, False otherwise
    """
    expand_path = os.path.expanduser(probe_path)  # expand ~ to usr home
    if base_path is None:
        base_path = os.getcwd()
    if os.path.isabs(expand_path) and os.path.exists(expand_path):
        return expand_path
    elif os.path.exists(os.path.join(base_path, probe_path)):
        return os.path.join(base_path, probe_path)
    else:
        return False


def get_item(node: cv2.FileNode, yaml_base_path: str):
    """get an item from a cv2.FileNode, recursively parse the item if it is a Map, List or path

    Args:
        node (cv2.FileNode): file node to be parsed
        yaml_base_path (str): the base path of the current yaml file

    Returns:
        the read content
    """
    if node.isNone():  # empty
        return None
    elif node.isMap():  # dict
        keys = node.keys()
        if all(mat_key in keys for mat_key in ["rows", "cols", "dt", "data"]):  # matrix
            return node.mat()
        else:  # key-value pairs
            dict = {}
            for key in keys:
                dict[key] = get_item(node.getNode(key), yaml_base_path)
            return dict
    elif node.isSeq():  # list
        list = []
        for i in range(node.size()):
            list.append(get_item(node.at(i), yaml_base_path))
        return list
    elif node.isReal() or node.isInt():  # number
        return node.real()
    elif node.isString():  # string
        path = parse_path(
            node.string(), yaml_base_path
        )  # try parsing the string as path
        if path:
            return read_file(path)
        else:  # not a path
            return node.string()


def read_file(path: str):
    """test the path, read a file, if it is a yaml file; parse it, if it is an image, read it; if it is a torchscript module, return the path; otherwise raise exception for file not supported.

    Args:
        path (str): path to the file, can be absolute or relative

    Returns:
        content of the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    if path.endswith(".yaml") or path.endswith(".yml"):
        yaml_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        return get_item(yaml_file.root(), os.path.dirname(path))
    elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return cv2.imread(path)
    elif path.endswith(".ts") or path.endswith(".trt"):
        return load_model(path)
    else:
        raise Exception(f"File {path} is not supported")
