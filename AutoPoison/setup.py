from setuptools import setup

setup(
    name="AutoPoison",
    version="0.0.1",
    # Single-package layout: code sits directly in AutoPoison/ root, not in
    # AutoPoison/AutoPoison/. Tell setuptools explicitly where each package
    # lives on disk, otherwise it expects AutoPoison/ to be a subdirectory
    # of the project root and fails with "package directory 'AutoPoison'
    # does not exist".
    packages=["AutoPoison", "AutoPoison.quant_specific"],
    package_dir={
        "AutoPoison": ".",
        "AutoPoison.quant_specific": "quant_specific",
    },
    install_requires=[],
)