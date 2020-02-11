# Master's Thesis: Christoffer Øhrstrøm
This repository contains all the code and documentation of the work done in Christoffer Øhrstrøms' master's thesis.

## Requirements
This work has only been tested on the following dependencies:

* g++
* Python v3.7.5

## Installation
First clone the project.

``` sh
git clone https://github.com/ChrisFugl/masters_thesis
cd masters_thesis
```

**Note:** We recommend that you install the Python packages in a virtual environment. See the next section for how to do this, and then proceed with the rest of this section afterwards.

``` sh
sh install.sh
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

## Usage
Different scripts are used to preprocess data, training, and evaluation. The scripts are configured using [Hydra](https://hydra.cc/). See default configurations in `configs/`. The following steps are required to preprocess the data where you have to add command line flags to point to your data.

1. Create oracle files: `python create_oracle.py`
2. Create clusters for generative RNNG: `python clustering.py`

You may optionally want to create artificial data for debugging purposes:

``` sh
python create_artificial_data.py
```

## Oracle Files
Oracle files are generated from annotated parse files. The format of an Oracle file is as follows:

```
brackets_1
actions_1
tokens_1
unknownified_tokens_1
...
brackets_N
actions_N
tokens_N
unknownified_tokens_N
```

The N annotated parses are transformed into (1) the original annotated parses (brackets_i), (2) the sequence of actions necessary to generate the parse tree (actions_i), (3) tokens in the annotated parse (tokens_i), and (4) tokens in the annotated tree where the tokens that only occurs one have been replaced with an unknown form (unknownified_tokens_i).

Use `create_oracle.py` to generate an oracle from an annotated parse file.

## Acknowledgements
The implementation of Brown Clustering is the work of [Percy Liang](https://github.com/percyliang/brown-cluster) and no changes have been made to his work. Permission is granted for anyone to copy, use, or modify these programs and accompanying documents for purposes of research or education, provided this copyright notice is retained, and note is made of any changes that have been made.
