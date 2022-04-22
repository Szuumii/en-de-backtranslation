import setuptools

# Parse requirements
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Get long description
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# Setup package
setuptools.setup(
    name="backtranslation_lab",
    version="1.0.0",
    author="Wiktor Łazarski & Jakub Szumski",
    author_email="wjlazarski@gmail.com",
    description="Backtranslation augmentation tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Szuumii/en-de-backtranslation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)
