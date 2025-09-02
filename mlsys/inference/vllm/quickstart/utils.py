#!/usr/bin/env python3

import pathlib

# get current directory
def _get_current_path():
    return pathlib.Path(__file__).parent.resolve()


def get_model_path(model_name: str) -> str:
    model_path = _get_current_path() / "models" / model_name
    return str(model_path)
