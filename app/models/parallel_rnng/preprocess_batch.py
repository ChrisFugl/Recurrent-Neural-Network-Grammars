from app.constants import (
    ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX, ACTION_NON_TERMINAL_INDEX, ACTION_REDUCE_INDEX,
    ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE, ACTION_NON_TERMINAL_TYPE,
    PAD_INDEX
)
import torch

def preprocess_batch(
        device,
        valid_args,
        non_terminal_converter,
        token_converter,
        batch_size,
        max_action_length,
        batch_actions,
        batch_tokens_lengths,
        batch_tokens_tensor,
        batch_tags_tensor
    ):
    """
    :type device: torch.device
    :type valid_args: (app.data.action_set.action_set.ActionSet, bool, int, int, int, int, int, int, int)
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
    token_counter = [0 for _ in range(batch_size)]
    last_action = [None for _ in range(batch_size)]
    open_nt_count = [0 for _ in range(batch_size)]
    actions_in_batch = enumerate(batch_actions)
    stack_size = [0 for _ in range(batch_size)]
    batch_list_tokens_length = [length.cpu().item() for length in batch_tokens_lengths]
    max_stack_size = 0
    for action_index in range(max_action_length):
        actions_in_batch, actions = get_actions_at_index(actions_in_batch, action_index)
        nt_index = [PAD_INDEX for _ in range(batch_size)]
        compose_nt_index = [PAD_INDEX for _ in range(batch_size)]
        token_index = [PAD_INDEX for _ in range(batch_size)]
        tag_index = [PAD_INDEX for _ in range(batch_size)]
        number_of_children = [0 for _ in range(batch_size)]
        invalid_mask = get_invalid_mask(device, valid_args, batch_list_tokens_length, token_counter, last_action, open_nt_count)
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
                token_counter[batch_index] = token_counter[batch_index] + 1
                stack_size[batch_index] = stack_size[batch_index] + 1
                parent.add_child(node)
            elif type == ACTION_GENERATE_TYPE:
                token_index[batch_index] = token_converter.token2integer(action.argument)
                token_counter[batch_index] = token_counter[batch_index] + 1
                stack_size[batch_index] = stack_size[batch_index] + 1
                parent.add_child(node)
            elif type == ACTION_NON_TERMINAL_TYPE:
                if parent is not None:
                    parent.add_child(node)
                nt_index[batch_index] = non_terminal_converter.non_terminal2integer(action.argument)
                open_nt_count[batch_index] = open_nt_count[batch_index] + 1
                stack_size[batch_index] = stack_size[batch_index] + 1
                parents[batch_index] = node
            else:
                compose_nt_index[batch_index] = non_terminal_converter.non_terminal2integer(parent.action.argument)
                number_of_children[batch_index] = len(parent.children)
                open_nt_count[batch_index] = open_nt_count[batch_index] - 1
                stack_size[batch_index] = stack_size[batch_index] - len(parent.children)
                parents[batch_index] = parent.parent
            max_stack_size = max(max_stack_size, stack_size[batch_index])
            last_action[batch_index] = action
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
            invalid_mask,
        )
        output.append(preprocessed)
    return output, max_stack_size

def get_actions_at_index(actions_in_batch, action_index):
    actions_in_batch = list(filter(lambda element: action_index < len(element[1]), actions_in_batch))
    actions = []
    for i, acts in actions_in_batch:
        actions.append((i, acts[action_index]))
    return actions_in_batch, actions

def get_invalid_mask(device, valid_args, tokens_length, token_counter, last_action, open_nt_count):
    action_set, generative, action_count, reduce_index, shift_index, gen_start, gen_count, nt_start, nt_count = valid_args
    batch_size = len(token_counter)
    mask = torch.ones((batch_size, action_count), device=device, dtype=torch.bool)
    for i in range(batch_size):
        valid_actions, _ = action_set.valid_actions(tokens_length[i], token_counter[i], last_action[i], open_nt_count[i])
        if ACTION_REDUCE_INDEX in valid_actions:
            mask[i, reduce_index] = 0
        if generative and ACTION_GENERATE_INDEX in valid_actions:
            mask[i, gen_start:gen_start + gen_count] = 0
        if not generative and ACTION_SHIFT_INDEX in valid_actions:
            mask[i, shift_index] = 0
        if ACTION_NON_TERMINAL_INDEX in valid_actions:
            mask[i, nt_start:nt_start + nt_count] = 0
    return mask

class Preprocessed:

    def __init__(self, actions_indices, non_terminal_index, compose_non_terminal_index, number_of_children, token_index, tag_index, invalid_mask):
        """
        :type actions_indices: list of (int, app.data.actions.action.Action)
        :type non_terminal_index: torch.Tensor
        :type compose_non_terminal_index: torch.Tensor
        :type number_of_children: torch.Tensor
        :token_index: torch.Tensor
        :tag_index: torch.Tensor
        :type invalid_mask: torch.Tensor
        """
        self.actions_indices = actions_indices
        self.non_terminal_index = non_terminal_index
        self.compose_non_terminal_index = compose_non_terminal_index
        self.token_index = token_index
        self.tag_index = tag_index
        self.number_of_children = number_of_children
        self.invalid_mask = invalid_mask

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
