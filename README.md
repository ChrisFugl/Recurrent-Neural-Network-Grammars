# Parallel Recurrent Neural Network Grammars
This repository contains the code associated with the paper: Parallel Recurrent Neural Network Grammars. This includes a PyTorch implementation of Recurrent Neural Network Grammars.

[TOC]

## Requirements
This work has only been tested on the following dependencies:

* g++ v7.4.0
* Python v3.7.5
* [Graphviz](http://graphviz.org/) v2.44.0 (for visualising trees)

## Installation
The project can be installed by running the following script from the root of the project. This assumes that you use a UNIX-based system and have already installed the requirements. 

``` sh
sh install.sh
```

For Windows users: We suggest that you run the Windows commands that correspond to the ones in install.sh.

pygraphviz might fail to install if Graphviz has not been properly installed. This depends on the system. For example, the `libgraphviz-dev` had to be installed on Ubuntu, before pygraphviz could be installed:

``` sh
apt install graphviz libgraphviz-dev
```

## Usage
Different scripts are used to preprocess data, training, sampling, and evaluation. The scripts are configured using [Hydra](https://hydra.cc/). See default configurations in `configs/`.

### Preprocessing

The following steps are required to preprocess the data, where you have to add command line flags to point to your data.

1. Create oracle files: `python create_oracle.py`
2. Create clusters for Generative RNNG: `python clustering.py`

You may optionally want to create artificial data for debugging purposes:

``` sh
python create_artificial_data.py
```

Note that you will have to call `create_oracle.py` and `clustering.py` on the output of `create_artificial_data.py`, in order to obtain oracle and cluster files for the artificial data.

### Training

Models can be trained using `train.py`. The following command can be used to train a Discriminative Parallel RNNG on the Wall Street Journal (WSJ) sections of the Penn English Treebank (PET), assuming that the data is located in `data/pet/wsj/discriminative/train.oracle`, and the word vectors are located in `data/pet/wsj/sskip.100.vectors`. 

``` sh
python train.py type=dis loader=wsj.dis.oracle model=dis.parallel_rnng name=wsj/dis/parallel_rnng
```

See `configs/train.yaml` for all configuration parameters and their defaults. See `configs/models/` for model specific configurations. Remember to set the `gpu` flag to specify which GPU to use. All scripts default to use the CPU, unless a GPU is specified.

``` sh
python train.py type=dis loader=wsj.dis.oracle model=dis.parallel_rnng gpu=0 name=<experiment name>
```

The log files, tensorboard output, and model parameters will be saved to `outputs/train/<experiment name>/`.

**Note**: The training process defaults to keep training, until stopped by the user. Enter `ctrl + c` to stop training. This will initiate a final evaluation of the model, followed by an exit. Press `ctrl + c` again to force exit immediately. You can change when to stop training by setting a different `stopping_criterion`.

### Sampling

Use `sample.py` to sample trees from discriminative models. Generative sampling is not supported. Sampling can either be done using greedy sampling

``` sh
python sample.py sampler=greedy sampler.load_dir=outputs/train/wsj/dis/parallel_rnng data=test gpu=0 name=<experiment name>
```

or ancestral sampling

``` sh
python sample.py sampler=ancestral sampler.load_dir=outputs/train/wsj/dis/parallel_rnng data=test gpu=0 name=<experiment name>
```

The samples will be saved to `outputs/sample/<experiment name>/samples.json`. Remember to set gpu option to increase sampling speed.

**Note**: `sample.py` accepts a data flag, which may either be *val* or *test* (defaults to *val*), depending on whether you want to sample from the validationset or testset, respectively.

### Evaluation

Evaluation is measured in F1 score for discriminative models, and also in word-level perplexity for generative models. The evaluation script automatically detects whether the model is discriminative or generative. 

``` sh
python evaluate.py samples=<path to samples.json> model_dir=outputs/train/wsj/dis/parallel_rnng gpu=0 name=<experiment name>
```

Output will be saved to `outputs/evaluate/<experiment name>/`.

### Statistics

#### Time Statistics

Use `time_stats.py` to obtain statistics about how many sequences a model can process per second.

``` sh
python time_stats.py type=dis loader=wsj.dis.oracle model=dis.parallel_rnng iterator.batch_size=64 gpu=0 name=<experiment name>
```

This process may take a while to finish, as the measurements are done on the trainingset.

Output is saved to `outputs/time_stats/<experiment name>/`.

#### Oracle Statistics

Use `oracle_stats.py` to get statistics about a oracle file, as generated by `create_oracle.py`.

``` sh
python oracle_stats.py loader=wsj.dis.oracle
```

## Structure

The most important parts of the codebase are mentioned here:

**app/models/**: RNNG and Parallel RNNG, as well as two LSTM models that are not relevant to the "Parallel Recurrent Neural Network Grammars" paper.

**app/samplers/**: Greedy and ancestral sampling.

**app/evaluators/**: Evaluation of discriminative models (F1) and generative models (F1 + word-level perplexity).

**app/data/**: Everything related to loading and preprocessing of files - both raw and oracle files. This directory also contains definitions of iterators (unordered and ordered), along with class representations of actions.

**configs/**: Contains all the configuration files related to each of the aforementioned tasks. The configuration files specify which arguments to accept along with their defaults. The configuration architecture is based on [Hydra](https://hydra.cc/).

## Oracle Files

Oracle files are generated from annotated parse files by using `create_oracle.py`. The format of an Oracle file is as follows:

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

The N annotated parses are transformed into: (1) the original annotated parses (brackets_i), (2) the sequence of actions necessary to generate the parse tree (actions_i), (3) tokens in the annotated parse (tokens_i), and (4) tokens in the annotated parse, where the tokens that only occurs once have been replaced with their unknownified equivalents (unknownified_tokens_i).

## Acknowledgements
The implementation of Brown Clustering is the work of [Percy Liang](https://github.com/percyliang/brown-cluster), and no changes have been made to his work. Permission is granted for anyone to copy, use, or modify these programs and accompanying documents for purposes of research or education, provided this copyright notice is retained, and note is made of any changes that have been made.
