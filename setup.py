"""Setup configuration for fn-dbscan package."""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
        include_package_data=True,
    )
