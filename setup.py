from setuptools import setup, find_packages

reqs = [
    "h5py",
    "jupyterlab",
    "matplotlib",
    "numpy",
    "pandas",
    "pillow",
    "scipy==1.2.1",
    "scikit-learn",
    "yaml",
    "imageio",
    "tensorflow",
    "keras",
    "pydot",
    "pytorch"
]

conda_reqs = [
    "h5py",
    "jupyterlab",
    "matplotlib",
    "numpy",
    "pandas",
    "pillow",
    "scipy==1.2.1",
    "scikit-learn",
    "yaml",
    "imageio",
    "tensorflow",
    "keras",
    "pydot",
    "pytorch"
]

test_pkgs = []

setup(
    name="geoimages",
    python_requires='>3.4',
    description="ML for geotechnical consulting firm.",
    url="https://github.com/neumj/geo-images",
    install_requires=reqs,
    conda_install_requires=conda_reqs,
    test_requires=test_pkgs,
    packages=find_packages(),
    include_package_data=True
)
