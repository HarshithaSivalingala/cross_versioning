from setuptools import setup, find_packages

setup(
    name="ml-repo-upgrader",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.8",
)