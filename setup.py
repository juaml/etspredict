from setuptools import find_packages, setup

# requirements = []
# with open("requirements.txt", "r") as f:
#    for line in f:
#        requirements.append(line)

setup(
    name="etspredict",
    version="0.1.0",
    description="code for edge time series + prediction project",
    url="https://github.com/juaml/etspredict",
    author="Applied Machine Learning FZJ",
    packages=find_packages(),
    # install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={"": ["data/*"]},
)
