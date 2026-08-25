from setuptools import setup, find_packages

setup(
    name="anna-neuromorphic",
    version="1.0.0",
    description="ANNA: Autonomous Neuromorphic Navigation Architecture - A Microwatt-Scale Spiking Brain and Synthesizable ASIC Architecture",
    author="ANNA Research and Development Group",
    author_email="research@anna-neuromorphic.org",
    url="https://github.com/Brayan114/Project-ANNA",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pytest>=7.0.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
