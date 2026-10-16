#!/usr/bin/env python3

from setuptools_scm import get_version

print(
    get_version(
        root=".",
        git_describe_command=["python3", "scripts/release/get_version_tag.py"],
    )
)
