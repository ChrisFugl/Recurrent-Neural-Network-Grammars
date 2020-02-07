# Master's Thesis: Christoffer Øhrstrøm
This repository contains all the code and documentation of the work done in Christoffer Øhrstrøms' master's thesis.

## Requirements
This work has only been tested on the following dependencies:

* Python v3.7.5

## Installation
First clone the project.

``` sh
git clone https://github.com/ChrisFugl/masters_thesis
cd masters_thesis
```

**Note:** We recommend that you install the Python packages in a virtual environment. See the next section for how to do this, and then proceed with the rest of this section afterwards.

``` sh
pip install -r requirements.txt
```

### Virtual Environment (optional)
A virtual environment helps you to avoid that Python packages in this project does not conflict with other Python packages in your system. Follow the instructions [on this site](https://virtualenv.pypa.io/en/stable/installation/) to install the virtualenv package, which enables you to create virtual environments.

Once virtualenv is installed, you will need to run the following commands to setup a virtual environment for this project.

``` sh
virtualenv env
```

You may want to add the flag "--python python3" in case your default Python interpreter is not at version 3 (run ```python --version``` to check the Python version):

``` sh
virtualenv --python python3 env
```

Either of the previous two commands will create a directory called *env* in the project directory. You need to run the following command to make use of the virtual environment.

``` sh
source env/bin/activate
```

You are now up an running with the virtual environment. Run the following command when you want to exit this environment.

``` sh
deactivate
```

### Jupyter Support (optional)
The commands in this section should be run from inside of a virtual environment. Note that you only need to do these steps if you are using a virtual environment.

Run this command:

``` sh
python -m ipykernel install --user --name=MscThesis
```

Use this command to start a Jupyter notebook:

``` sh
jupyter notebook
```

Select "MscThesis" when creating a new notebook.
