from setuptools import setup, find_packages

setup(
    name='echostate',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "optuna",
        "pandas",
        "pytest",
        "qutip",
        "safetensors",
        "scipy",
        "setuptools",
        "torch",
    ],
    description='Echo State Network implementation using PyTorch',
    author='Alexander Belik',
    license='',
)
