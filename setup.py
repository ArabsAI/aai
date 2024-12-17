from setuptools import setup

setup(
    name="aai",
    packages=["aai"],
    entry_points={
        "console_scripts": [
            "aai=aai.aai:main",
        ],
    },
)
