# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    # https://packaging.python.org/specifications/core-metadata/#name
    name='trainprocgen',

    # https://www.python.org/dev/peps/pep-0440/
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0',

    packages=find_packages(),

    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.5, <4',

    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas', 'pyyaml', 'torch', 'numpy', 'gym'],
)
