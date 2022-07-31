# Introduction

[A sample Pytest module for a Scikit-learn model training function](https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Pytest) used a basis for adding [pre-commit](https://pre-commit.com/) configuration.

## Requirements

- git
- python
- pip

## Testing

Windows

````powershell

git clone https://github.com/tonyskidmore/python-pre-commit-example.git
cd python-pre-commit-example
python -m venv .venv
.\.venv\Scripts\activate
pip install pip --upgrade
pip install setuptools --upgrade
pip install -r requirements.txt
pip install -r dev-requirements.txt
pytest

````

## pre-commit

````powershell

# install pre-commit and other libraries
pip install -r dev-requirements

````
