import setuptools

setuptools.setup(
    name="modelwhaleutils",
    version="0.5.2.1",
    author="modalwhale team",
    description="use in mw",
    url="https://github.com/Kesci/modelwhaleutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["PyJWT", "requests",
                      "pynvml", "psutil", "boto3", "mlflow==2.2.0"],
    python_requires='>=3.6',
)
