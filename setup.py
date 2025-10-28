from setuptools import setup, find_packages

setup(
    name="production_game",
    version="0.1.0",
    description="A multi-agent game simulation with Farmer, Consumer, and Government",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "gym>=0.18.0",
        "matplotlib>=3.4.0",
        "tensorboard>=2.6.0",
    ],
)
