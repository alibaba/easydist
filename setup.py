# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import setuptools
from importlib.machinery import SourceFileLoader

version = (
    SourceFileLoader("metadist.version", os.path.join(
        "metadist", "version.py")).load_module().VERSION
)


def is_comment_or_empty(line):
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def remove_comments_and_empty_lines(lines):
    return [line for line in lines if not is_comment_or_empty(line)]


def get_core_requirements():
    with open(os.path.join("requirements", "core-requirements.txt")) as f:
        core_requirements = remove_comments_and_empty_lines(
            f.read().splitlines())
    return core_requirements


def get_long_description():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


setuptools.setup(
    name="metadist",
    version=version,
    author="Shenggan Cheng",
    author_email="shenggan.c@u.nus.edu",
    description="Efficient Automatic Training System for Super-Large Models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/alibaba/easydist",
    packages=setuptools.find_packages(),
    install_requires=get_core_requirements(),
    extras_require={
        "torch": [
            "torch",
            "torchvision",
        ],
        "jax": [
            "jax[cuda11_pip]",
            "flax",
        ]
    },
)
