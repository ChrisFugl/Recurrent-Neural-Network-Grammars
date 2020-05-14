from app.constants import ACTION_SHIFT_TYPE, ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_GENERATE_TYPE
from app.data.actions.non_terminal import NonTerminalAction
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_rnn_parser_attention(batch, attention_weights):
    """
    :type batch: app.data.batch.Batch
    :type attention_weights: list of torch.Tensor
    """
    n_actions = batch.actions.lengths[0].cpu().item()
    n_tokens = batch.tokens.lengths[0].cpu().item()
    attention_weights = [weights[0, :n_tokens] for weights in attention_weights[:n_actions]]
    weights = torch.stack(attention_weights, dim=0)
    weights_np = weights.detach().cpu().numpy().transpose()
    fig = plt.figure()
    ax = plt.gca()
    ylabels = list(reversed(batch.tokens.tokens[0]))
    yticks = np.arange(0, n_tokens, 1)
    yticks_minor = np.arange(-0.5, n_tokens + 0.5, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.xticks(rotation=90)
    xlabels = list(map(str, batch.actions.actions[0]))
    xticks = list(np.arange(0, n_actions, 1))
    xticks_minor = np.arange(-0.5, n_actions + 0.5, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(which='minor', color='#dadada', linestyle='-', linewidth=1)
    plt.imshow(weights_np, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    return fig

def visualize_buffer_attention(batch, attention_weights):
    """
    :type batch: app.data.batch.Batch
    :type attention_weights: list of torch.Tensor
    """
    n_actions = batch.actions.lengths[0].cpu().item()
    n_tokens = batch.tokens.lengths[0].cpu().item()
    max_length = n_tokens + 1
    expanded = []
    for i in range(n_actions):
        weights = attention_weights[i]
        tensor = torch.zeros((max_length,))
        weights_length = min(max_length, weights.size(1))
        tensor[:weights_length] = weights[0, :weights_length]
        expanded.append(tensor)
    weights = torch.stack(expanded, dim=0).squeeze()
    weights_np = weights.detach().cpu().numpy().transpose()
    fig = plt.figure()
    ax = plt.gca()
    ylabels = ['<SOS>', *(reversed(batch.tokens.tokens[0]))]
    yticks = np.arange(0, n_tokens + 1, 1)
    yticks_minor = np.arange(-0.5, n_tokens + 1.5, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.xticks(rotation=90)
    xlabels = list(map(str, batch.actions.actions[0]))
    xticks = list(np.arange(0, n_actions, 1))
    xticks_minor = np.arange(-0.5, n_actions + 0.5, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(which='minor', color='#dadada', linestyle='-', linewidth=1)
    plt.imshow(weights_np, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    return fig

def visualize_history_attention(batch, attention_weights):
    """
    :type batch: app.data.batch.Batch
    :type attention_weights: list of torch.Tensor
    """
    n_actions = batch.actions.lengths[0].cpu().item()
    max_length = n_actions
    expanded = []
    for i in range(max_length):
        weights = attention_weights[i]
        tensor = torch.zeros((max_length,))
        weights_length = min(max_length, weights.size(1))
        tensor[:weights_length] = weights[0, :weights_length]
        expanded.append(tensor)
    weights = torch.stack(expanded, dim=0).squeeze()
    weights_np = weights.detach().cpu().numpy().transpose()
    fig = plt.figure()
    ax = plt.gca()
    ylabels = ['<SOS>', *map(str, batch.actions.actions[0][:-1])]
    yticks = np.arange(0, n_actions, 1)
    yticks_minor = np.arange(-0.5, n_actions + 1.5, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.xticks(rotation=90)
    xlabels = list(map(str, batch.actions.actions[0]))
    xticks = list(np.arange(0, n_actions, 1))
    xticks_minor = np.arange(-0.5, n_actions + 0.5, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(which='minor', color='#dadada', linestyle='-', linewidth=1)
    plt.imshow(weights_np, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    return fig

def visualize_stack_buffer_attention(batch, attention_weights):
    """
    :type batch: app.data.batch.Batch
    :type attention_weights: list of torch.Tensor
    """
    n_actions = batch.actions.lengths[0].cpu().item()
    attention_weights = [weights[0] for weights in attention_weights[:n_actions]]
    weights = torch.stack(attention_weights, dim=0)
    weights_np = weights.detach().cpu().numpy().transpose()
    fig = plt.figure()
    ax = plt.gca()
    ylabels = ['S', 'B']
    n_structures = len(ylabels)
    yticks = np.arange(0, n_structures, 1)
    yticks_minor = np.arange(-0.5, n_structures + 1.5, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.xticks(rotation=90)
    xlabels = list(map(str, batch.actions.actions[0]))
    xticks = list(np.arange(0, n_actions, 1))
    xticks_minor = np.arange(-0.5, n_actions + 0.5, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(which='minor', color='#dadada', linestyle='-', linewidth=1)
    plt.imshow(weights_np, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    return fig

def visualize_stack_attention(batch, attention_weights):
    """
    :type batch: app.data.batch.Batch
    :type attention_weights: list of torch.Tensor
    """
    tree = []
    index_maps = []
    stack = []
    shift_index = 0
    for i, action in enumerate(batch.actions.actions[0], start=1):
        index_map = []
        for _, interval in stack:
            index_map.append(interval)
        index_maps.append(index_map)
        type = action.type()
        if type == ACTION_NON_TERMINAL_TYPE:
            tree_string = f'({action.argument}'
            stack.append((action, (i, i + 1)))
        elif type == ACTION_SHIFT_TYPE:
            tree_string = batch.tokens.tokens[0][shift_index]
            shift_index += 1
            stack.append((action, (i, i + 1)))
        elif type == ACTION_GENERATE_TYPE:
            tree_string = action.argument
            stack.append((action, (i, i + 1)))
        elif type == ACTION_REDUCE_TYPE:
            tree_string = ')'
            while True:
                child, interval = stack.pop()
                if child.type() == ACTION_NON_TERMINAL_TYPE and child.open:
                    break
            start, _ = interval
            nt_action = NonTerminalAction(child.argument, open=False)
            stack.append((nt_action, (start, i + 1)))
        else:
            raise Exception(f'Unknown action type: {type}')
        tree.append(tree_string)
    max_length = len(tree)
    n_actions = batch.actions.lengths[0].cpu().item()
    expanded = []
    for i in range(n_actions):
        weights = attention_weights[i]
        tensor = torch.zeros((max_length,))
        index_map = index_maps[i]
        tensor[0] = weights[0, 0]
        for j, (start, end) in enumerate(index_map, start=1):
            tensor[start:end] = weights[0, j]
        expanded.append(tensor)
    weights = torch.stack(expanded, dim=0).squeeze()
    weights_np = weights.detach().cpu().numpy().transpose()
    fig = plt.figure()
    ax = plt.gca()
    ylabels = ['<SOS>', *tree]
    yticks = np.arange(0, len(tree) + 1, 1)
    yticks_minor = np.arange(-0.5, len(tree) + 1.5, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    plt.xticks(rotation=90)
    xlabels = list(map(str, batch.actions.actions[0]))
    xticks = list(np.arange(0, n_actions, 1))
    xticks_minor = np.arange(-0.5, n_actions + 0.5, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(which='minor', color='#dadada', linestyle='-', linewidth=1)
    plt.imshow(weights_np, cmap='Greys', aspect='equal', vmin=0, vmax=1)
    return fig
