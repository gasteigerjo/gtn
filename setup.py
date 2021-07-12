from setuptools import setup

install_requires_gtn = [
    "numpy",
    "scipy>=1.3",
    "torch==1.4",  # norm broken in 1.5, some torchscript stuff broken later (v1.8, 1.9)
    "torch_geometric",
    "scikit-learn",
    "tensorboard",
    "networkx",
    "sacred",
    "seml",
    "lcn",  # See https://github.com/klicperajo/lcn
]

setup(
    name="GTN",
    version="1.0",
    description="Graph Transport Network",
    author="Johannes Klicpera",
    author_email="klicpera@in.tum.de",
    packages=["gtn"],
    install_requires=install_requires_gtn,
    zip_safe=False,
)
