import matplotlib.pyplot as plt
import numpy as np

def visualize_action_probs(actions, probs):
    """
    :type actions: list of app.data.actions.action.Action
    :type probs: list of float
    """
    fig = plt.figure(figsize=(len(actions), 6))
    x = np.arange(len(actions))
    names = [str(action) for action in actions]
    plt.bar(x, probs, alpha=0.5, color='c')
    plt.xticks(range(len(names)), names, rotation='vertical')
    plt.xlim(-0.5, len(actions) - 0.5)
    plt.grid(True)
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.ylim(bottom=0)
    plt.tight_layout()
    return fig
