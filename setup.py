#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="wificsi",
    version="0.0.1",
    description="WiFi Channel State Information extraction",
    author="Erick Rosete Beas",
    author_email="erickrosetebeas@hotmail.com",
    url="https://github.com/ErickRosete/wificsi",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "lightning==2.0.0",
        "torchmetrics==0.11.4",
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "hydra-optuna-sweeper==1.2.0",
        "wandb",
        "pre-commit",  # hooks for applying linters on commit
        "rich",  # beautiful text formatting in terminal
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
