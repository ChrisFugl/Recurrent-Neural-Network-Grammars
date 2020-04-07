from app.constants import ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

NODE_COLOR = [[0, 0, 0, 0]]
FONT_COLOR = 'blue'
FONT_SIZE = 9

def visualize_tree(actions, tokens):
    """
    :type actions: list of app.data.actions.action.Action
    :type tokens: list of str
    """
    fig = plt.figure()
    graph = make_graph(actions, tokens)
    pos = graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=True, arrows=False, node_color=NODE_COLOR, font_size=FONT_SIZE, font_color=FONT_COLOR)
    return fig

def make_graph(actions, tokens):
    graph = nx.DiGraph()
    token_index = 0
    node_id = 0
    parents = []
    parent_label = None
    for action in actions:
        type = action.type()
        if type == ACTION_NON_TERMINAL_TYPE:
            node_label = f'{node_id}:{action.argument}'
            graph.add_node(node_label)
            if parent_label is not None:
                graph.add_edge(parent_label, node_label)
            parents.append(node_label)
            parent_label = node_label
            node_id += 1
        elif type == ACTION_SHIFT_TYPE:
            node_label = f'{node_id}:{tokens[token_index]}'
            graph.add_node(node_label)
            graph.add_edge(parent_label, node_label)
            token_index += 1
            node_id += 1
        elif type == ACTION_GENERATE_TYPE:
            node_label = f'{node_id}:{action.argument}'
            graph.add_node(node_label)
            graph.add_edge(parent_label, node_label)
            node_id += 1
        else: # reduce
            parents.pop()
            if len(parents) == 0:
                parent_label = None
            else:
                parent_label = parents[-1]
    return graph
