# Synthetic, trimmed excerpt of setup.py.
from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    # Requirements written as git+https install URLs (e.g. "pkg @ git+https://...")
    # are passed through to install_requires unchanged.
    return lines


setup(
    name="tensorrt_llm",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)
