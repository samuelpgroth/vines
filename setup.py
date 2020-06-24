import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vines", # Replace with your own username
    version="0.0.1",
    author="Samuel Groth",
    author_email="samuelpgroth@gmail.com",
    description="Volume INtegral Equation Solver (VINES)",
    url="https://github.com/samuelpgroth/vines",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)