import setuptools


def readme():
    with open("README.md", "r") as f:
        return f.read()


setuptools.setup(
    name="vines",
    version="0.0.1",
    author="Samuel Groth",
    author_email="samuelpgroth@gmail.com",
    description="Volume INtegral Equation Solver (VINES)",
    long_description=readme(),
    long_description_content_type='text/markdown',
    url="https://github.com/samuelpgroth/vines",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'':'vines'},
    python_requires='>=3.6',
    install_requires=['pyfftw','numba'],
    include_package_data=True
)