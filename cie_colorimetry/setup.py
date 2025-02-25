from setuptools import setup, find_packages

setup(
    name="cie_colorimetry",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",  # Added matplotlib dependency
    ],
    package_data={
        'cie_colorimetry': ['data/*.csv'],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for calculating CIE colorimetry parameters from emission spectra",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cie_colorimetry",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)