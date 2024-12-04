from setuptools import setup, find_packages

setup(
    name="quantum_compilation",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[ 
                      pgx,
                      jax,
                      qujax,
    ],
)
