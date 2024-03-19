from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = 'DimensionAnalysis',
    version = '0.1',
    packages=find_packages(),
    description='Utilizing Dimensional Analysis technique, which is commonly used in Physics , alongside SymbolicGPT to detect equations describing the AI-Feynman Datasets.',
    author='Amin Ravanbakhsh',
    author_email='amin.ravanbakhsh@uwaterloo.ca',
    url='https://aminravanbakhsh.com/',

    install_requires=required,
    python_requires='>=3.6',
)
