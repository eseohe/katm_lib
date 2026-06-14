"""Setup script for KATM — Keyphrase Anchored Topic Modeling."""

from setuptools import setup, find_packages

setup(
    name="katm",
    version="0.1.0",
    author="Eustace Ebhotemhen",
    description="Keyphrase Anchored Topic Modeling (KATM)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "nltk",
        "keybert",
        "yake",
        "rake-nltk",
        "spacy",
        "gensim",
        "pandas",
        "matplotlib",
        "jupyter",
        "notebook",
        "octis",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)