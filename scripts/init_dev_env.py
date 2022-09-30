#!/usr/bin/env python3

import os

from utils import exec_cmd, read_env


proj_path = read_env()["ROOT"].rstrip('/')
home_path = read_env()["HOME"].rstrip('/')

for item in os.listdir(os.path.join(proj_path, "env")):
    new_name = item.replace("env_", ".")
    src = os.path.join(proj_path, "env", item)
    dst = os.path.join(home_path, new_name)
    cmd = ["cp", "-n", src, dst]
    exec_cmd(cmd, for_real=True, print_cmd=True)
