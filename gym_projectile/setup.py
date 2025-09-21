from setuptools import setup

setup(
    name='gym_projectile',
    version='0.0.1',
    install_requires=['gymnasium', 'numpy', 'matplotlib'],
    packages=['gym_projectile', 'gym_projectile.envs'],
)
