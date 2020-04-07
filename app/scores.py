from app.constants import (
    ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE,
    EVALB_TOOL_PATH, EVALB_PARAMS_PATH
)
import hydra
import os
import re
import subprocess

EVALB_RECALL = 'Bracketing Recall'
EVALB_PRECISION = 'Bracketing Precision'
EVALB_F1 = 'Bracketing FMeasure'

def scores_from_samples(samples):
    """
    :type samples: list of app.samplers.sample.Sample
    :rtype: (float, float, float), str, str, str
    """
    tool_path = hydra.utils.to_absolute_path(EVALB_TOOL_PATH)
    params_path = hydra.utils.to_absolute_path(EVALB_PARAMS_PATH)
    gold_path = save_trees_to_brackets('gold', samples)
    predicted_path = save_trees_to_brackets('predicted', samples)
    process = subprocess.run([tool_path, '-p', params_path, gold_path, predicted_path], capture_output=True)
    evalb_output = str(process.stdout, 'utf-8')
    lines = evalb_output.split('\n')
    f1 = get_f1(lines)
    precision = get_precision(lines)
    recall = get_recall(lines)
    scores = (f1, precision, recall)
    return scores, evalb_output, gold_path, predicted_path

def save_trees_to_brackets(type, samples):
    """
    :type type: str
    :type samples: list of app.samplers.sample.Sample
    """
    # name, filename, trees, trees_tokens, trees_tags
    tokens = [sample.gold.tokens for sample in samples]
    tags = [sample.gold.tags for sample in samples]
    if type == 'gold':
        filename = 'trees.gld'
        trees = [sample.gold.actions for sample in samples]
    else:
        filename = 'trees.tst'
        trees = [sample.prediction.actions for sample in samples]
    path = get_path(filename)
    brackets = trees2brackets(trees, tokens, tags)
    content = '\n'.join(brackets)
    with open(path, 'w') as file:
        file.write(content)
    return path

def trees2brackets(trees, trees_tokens, trees_tags):
    brackets = []
    for tree, tokens, tags in zip(trees, trees_tokens, trees_tags):
        tree_brackets = []
        tag_index = 0
        token_index = 0
        for action in tree:
            type = action.type()
            if type == ACTION_NON_TERMINAL_TYPE:
                tree_brackets.append(f' ({action.argument}')
            elif type == ACTION_REDUCE_TYPE:
                tree_brackets.append(')')
            elif type == ACTION_SHIFT_TYPE:
                tree_brackets.append(f' ({tags[tag_index]} {tokens[token_index]})')
                tag_index += 1
                token_index += 1
            elif type == ACTION_GENERATE_TYPE:
                tree_brackets.append(f' ({tags[tag_index]} {action.argument})')
                tag_index += 1
            else:
                raise Exception(f'Unknown action: {type}')
        tree_brackets_string = ''.join(tree_brackets).strip()
        brackets.append(tree_brackets_string)
    return brackets

def get_path(filename):
    working_dir = os.getcwd()
    path = os.path.join(working_dir, filename)
    return path

def get_recall(lines):
    return get_bracket_score(EVALB_RECALL, lines)

def get_precision(lines):
    return get_bracket_score(EVALB_PRECISION, lines)

def get_f1(lines):
    return get_bracket_score(EVALB_F1, lines)

def get_bracket_score(name, lines):
    expression = re.compile(f'{name}\\s*=\\s*(\\d+(\\.\\d*))')
    for line in lines:
        match = expression.match(line)
        if match is not None:
            return float(match.group(1))
    return 0.0
