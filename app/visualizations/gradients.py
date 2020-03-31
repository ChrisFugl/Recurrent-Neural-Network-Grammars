import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def visualize_gradients(named_parameters):
    return gradient_flow(named_parameters, default_parameter_condition)

def visualize_gradients_exclude_representation(named_parameters):
    return gradient_flow(named_parameters, exclude_representation_condition)

def visualize_gradients_compose_only(named_parameters):
    return gradient_flow(named_parameters, compose_only_condition)

def gradient_flow(named_parameters, parameter_condition):
    '''
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    fig = plt.figure(figsize=(9, 6))
    for n, p in named_parameters:
        if parameter_condition(n, p):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color='c')
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color='b')
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation='vertical')
    plt.xlim(left=-0.5, right=len(ave_grads) - 0.5)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.ylabel('gradients')
    plt.grid(True)
    plt.legend([Line2D([0], [0], color='c', lw=4), Line2D([0], [0], color='b', lw=4)], ['max-gradient', 'mean-gradient'])
    plt.tight_layout()
    return fig

def default_parameter_condition(name, parameter):
    return (parameter.requires_grad and parameter.grad is not None) and ('bias' not in name)

def exclude_representation_condition(name, parameter):
    return (parameter.requires_grad and parameter.grad is not None) and ('bias' not in name) and (not name.startswith('representation'))

def compose_only_condition(name, parameter):
    return (parameter.requires_grad and parameter.grad is not None) and ('compose' in name)
