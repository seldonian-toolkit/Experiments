import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seldonian_experiments",
    version="0.0.5",
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
        "seldonian-engine>=0.2.2",
        "fairlearn==0.7.0",       
        "tqdm==4.64.0",
    ],
    packages=["experiments"],
    python_requires=">=3.8",
)