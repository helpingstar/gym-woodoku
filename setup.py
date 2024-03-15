from setuptools import setup, find_packages

setup(
    name="gym-woodoku",
    version="0.1.0",
    author="helpingstar",
    author_email="iamhelpingstar@gmail.com",
    description="It is a reinforcement learning environment for block puzzle games.",
    license="MIT License",
    packages=find_packages(),
    install_requires=["numpy>=1.26.4,<2.0.0", "pygame>=2.1.0,<3.0.0", "gymnasium>=1.0.0a1,<2.0.0a0"],
    python_requires=">=3.9,<4.0",
)
