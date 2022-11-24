from setuptools import setup, find_packages

setup(
    name="metrecs",
    version="0.0.1",
    license="MIT",
    author="",
    author_email="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/gabriben/metrecs",
    keywords="Recommender Systems",
    install_requires=[
        "numpy",
    ],
)
