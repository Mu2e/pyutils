from setuptools import setup

setup(
    name="pyutils",
    version="1.4.0",
    author="Sophie Middleton, Samuel Grant, Andrew Edmonds, Leo Borrel",
    description="Python tools for Mu2e collaborators",
    url="https://github.com/Mu2e/pyutils",
    packages=["pyutils"],
    package_data={"pyutils": ["mu2e.mplstyle"]},  
    include_package_data=True,
)
