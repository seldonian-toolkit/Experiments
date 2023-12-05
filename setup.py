import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seldonian_experiments",
    version="0.2.4",
    author="Austin Hoag",
    author_email="austinthomashoag@gmail.com",
    description="Library for running experiments with Seldonian algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="",
    project_urls={
        "Bug Tracker": "https://github.com/seldonian-framework/Experiments/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "seldonian-engine",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)