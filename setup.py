from setuptools import setup, find_packages

setup(
    name="ingacfair",
    version="0.1",
    author="Anil Panda",
    author_email="anilkumar.panda@ing.com",
    description="INGA Connect fairness workshop",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3.11"
    ],
    python_requires=">=3.11",
    install_requires=[
        # List your project dependencies here
    ]
)