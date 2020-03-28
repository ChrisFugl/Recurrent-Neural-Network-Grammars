from app.constants import ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE, PAD_INDEX

import torch

def preprocess_batch(device, non_terminal_converter, token_converter, batch_size, max_action_length, batch_actions, batch_tokens_lengths, batch_tokens_tensor, batch_tags_tensor):
    """
    :type device: torch.device
    :type non_terminal_converter: app.data.converters.non_terminal.NonTerminalConverter
    :type token_converter: app.data.converters.token.TokenConverter
    :type batch_size: int
    :type max_action_length: int
    :type batch_actions: list of list of app.data.actions.action.Action
    :type batch_tokens_lengths: torch.Tensor
    :type batch_tokens_tensor: torch.Tensor
    :type batch_tags_tensor: torch.Tensor
    :rtype: list of app.models.parallel_rnng.preprocess_batch.Preprocessed, int
    """
    shift_indices = [0 for _ in range(batch_size)]
    parents = [None for _ in range(batch_size)]
    output = []
    actions_in_batch = enumerate(batch_actions)
    stack_size = [0 for _ in range(batch_size)]
    max_stack_size = 0
    for action_index in range(max_action_length):
        actions_in_batch, actions = get_actions_at_index(actions_in_batch, action_index)
        nt_index = [PAD_INDEX for _ in range(batch_size)]
        compose_nt_index = [PAD_INDEX for _ in range(batch_size)]
        token_index = [PAD_INDEX for _ in range(batch_size)]
        tag_index = [PAD_INDEX for _ in range(batch_size)]
        number_of_children = [0 for _ in range(batch_size)]
        for batch_index, action in actions:
            type = action.type()
            parent = parents[batch_index]
            node = Tree(action, parent=parent)
            if type == ACTION_SHIFT_TYPE:
                shift_index = shift_indices[batch_index]
                # token buffer processes terminals in reverse order
                tokens_length = batch_tokens_lengths[batch_index]
                shift_tensor_index = tokens_length - shift_index - 1
                token_index[batch_index] = batch_tokens_tensor[shift_tensor_index, batch_index].cpu().item()
                tag_index[batch_index] = batch_tags_tensor[shift_tensor_index, batch_index].cpu().item()
                shift_indices[batch_index] = shift_index + 1
                stack_size[batch_index] = stack_size[batch_index] + 1
                parent.add_child(node)
            elif type == ACTION_GENERATE_TYPE:
                token_index[batch_index] = token_converter.token2integer(action.argument)
                stack_size[batch_index] = stack_size[batch_index] + 1
                parent.add_child(node)
            elif type == ACTION_NON_TERMINAL_TYPE:
                if parent is not None:
                    parent.add_child(node)
                nt_index[batch_index] = non_terminal_converter.non_terminal2integer(action.argument)
                stack_size[batch_index] = stack_size[batch_index] + 1
                parents[batch_index] = node
            else:
                compose_nt_index[batch_index] = non_terminal_converter.non_terminal2integer(parent.action.argument)
                number_of_children[batch_index] = len(parent.children)
                stack_size[batch_index] = stack_size[batch_index] - len(parent.children)
                parents[batch_index] = parent.parent
            max_stack_size = max(max_stack_size, stack_size[batch_index])
        nt_index_tensor = torch.tensor(nt_index, device=device, dtype=torch.long)
        compose_nt_index_tensor = torch.tensor(compose_nt_index, device=device, dtype=torch.long)
        token_index_tensor = torch.tensor(token_index, device=device, dtype=torch.long)
        tag_index_tensor = torch.tensor(tag_index, device=device, dtype=torch.long)
        number_of_children_tensor = torch.tensor(number_of_children, device=device, dtype=torch.long)
        preprocessed = Preprocessed(
            actions,
            nt_index_tensor,
            compose_nt_index_tensor,
            number_of_children_tensor,
            token_index_tensor,
            tag_index_tensor,
        )
        output.append(preprocessed)
    return output, max_stack_size

def get_actions_at_index(actions_in_batch, action_index):
    actions_in_batch = list(filter(lambda element: action_index < len(element[1]), actions_in_batch))
    actions = []
    for i, acts in actions_in_batch:
        actions.append((i, acts[action_index]))
    return actions_in_batch, actions

class Preprocessed:

    def __init__(self, actions_indices, non_terminal_index, compose_non_terminal_index, number_of_children, token_index, tag_index):
        """
        :type actions_indices: list of (int, app.data.actions.action.Action)
        :type non_terminal_index: torch.Tensor
        :type compose_non_terminal_index: torch.Tensor
        :type number_of_children: torch.Tensor
        :token_index: torch.Tensor
        :tag_index: torch.Tensor
        """
        self.actions_indices = actions_indices
        self.non_terminal_index = non_terminal_index
        self.compose_non_terminal_index = compose_non_terminal_index
        self.token_index = token_index
        self.tag_index = tag_index
        self.number_of_children = number_of_children

class Tree:

    def __init__(self, action, parent=None, children=None):
        self.action = action
        self.parent = parent
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)
