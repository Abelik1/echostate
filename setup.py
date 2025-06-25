from setuptools import setup, find_packages

setup(
    name='echostate',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'
    ],
    description='Echo State Network implementation using PyTorch',
    author='Alexander Belik',
    license='',
)
