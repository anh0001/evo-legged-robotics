from setuptools import setup, find_packages

setup(
    name="evo_legged_robotics",
    version="0.1.0",
    description="Evolutionary algorithms for legged robot locomotion in PyBullet",
    author="Anhar Risnumawan",
    author_email="risnumawan-anhar@ed.tmu.ac.jp",
    url="https://github.com/anh0001/evo-legged-robotics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybullet>=3.2.0",
        "tensorflow>=2.4.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
        ]
    },
)