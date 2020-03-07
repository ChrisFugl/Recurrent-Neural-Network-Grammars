from app.constants import (
    ACTION_REDUCE_INDEX, ACTION_NON_TERMINAL_INDEX, ACTION_SHIFT_INDEX, ACTION_GENERATE_INDEX,
    ACTION_REDUCE_TYPE, ACTION_NON_TERMINAL_TYPE, ACTION_SHIFT_TYPE, ACTION_GENERATE_TYPE,
)
import torch

def call_action(type, args):
    """
    :type type: int
    :type args: ActionArgs
    :rtype: torch.Tensor, int, int, int
    :returns: action log prob, open non-terminals count, token index, token count
    """
    function = _action2function[type]
    return function(args)

def reduce(args):
    """
    :type args: ActionArgs
    :rtype: torch.Tensor, int, int, int
    :returns: action log prob, open non-terminals count, token index, token count
    """
    stack = args.stack
    popped_items = []
    action = None
    while action is None or action.type() != ACTION_NON_TERMINAL_TYPE:
        state, action = stack.pop()
        popped_items.append(state)
    popped_tensor = torch.cat(popped_items, dim=0)
    action.close()
    non_terminal_embedding, _ = _get_non_terminal_embedding(args.embeddings.non_terminal_compose, action)
    composed = args.functions.composer(non_terminal_embedding, popped_tensor)
    stack.push(composed, action)
    outputs = args.outputs
    action_log_prob = _get_base_log_prop(args, ACTION_REDUCE_INDEX)
    return action_log_prob, outputs.open_non_terminals_count - 1, outputs.token_index, outputs.token_counter

def non_terminal(args):
    """
    :type args: ActionArgs
    :rtype: torch.Tensor, int, int, int
    :returns: action log prob, open non-terminals count, token index, token count
    """
    functions = args.functions
    non_terminal_embedding, argument_index = _get_non_terminal_embedding(args.embeddings.non_terminal, args.action)
    args.stack.push(non_terminal_embedding, args.action)
    outputs = args.outputs
    if args.log_probs is None:
        action_log_prob = None
    else:
        non_terminal_log_prob = _get_base_log_prop(args, ACTION_NON_TERMINAL_INDEX)
        non_terminal_logits = functions.representation2non_terminal_logits(args.log_probs.representation)
        conditional_non_terminal_log_probs = functions.logits2log_prob(non_terminal_logits)
        conditional_non_terminal_log_prob = conditional_non_terminal_log_probs[:, :, argument_index]
        action_log_prob = non_terminal_log_prob + conditional_non_terminal_log_prob
    return action_log_prob, outputs.open_non_terminals_count + 1, outputs.token_index, outputs.token_counter

def shift(args):
    """
    :type args: ActionArgs
    :rtype: torch.Tensor, int, int, int
    :returns: action log prob, open non-terminals count, token index, token count
    """
    outputs = args.outputs
    embedding = args.tokens_embedding[outputs.token_index, args.batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
    args.stack.push(embedding, args.action)
    action_log_prob = _get_base_log_prop(args, ACTION_SHIFT_INDEX)
    token_index = max(outputs.token_index - 1, 1)
    return action_log_prob, outputs.open_non_terminals_count, token_index, outputs.token_counter + 1

def generate(args):
    """
    :type args: ActionArgs
    :rtype: torch.Tensor, int, int, int
    :returns: action log prob, open non-terminals count, token index, token count
    """
    outputs = args.outputs
    stack = args.stack
    action_embedding = args.tokens_embedding[outputs.token_index, args.batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
    stack.push(action_embedding, args.action)
    if args.log_probs is None:
        action_log_prob = None
    else:
        generate_log_prob = _get_base_log_prop(args, ACTION_GENERATE_INDEX)
        token_log_prob = args.functions.token_distribution.log_prob(args.log_probs.representation, args.action.argument)
        action_log_prob = generate_log_prob + token_log_prob
    token_index = min(outputs.token_index + 1, args.tokens_length)
    return action_log_prob, outputs.open_non_terminals_count, token_index, outputs.token_counter + 1

def _get_non_terminal_embedding(embeddings, action):
    argument_index = action.argument_index_as_tensor()
    non_terminal_embedding = embeddings(argument_index).unsqueeze(dim=0).unsqueeze(dim=0)
    return non_terminal_embedding, argument_index

def _get_base_log_prop(args, action_index):
    if args.log_probs is None:
        return None
    else:
        return args.log_probs.log_prob_base[:, :, args.log_probs.action2index[action_index]]

_action2function = {
    ACTION_REDUCE_TYPE: reduce,
    ACTION_NON_TERMINAL_TYPE: non_terminal,
    ACTION_SHIFT_TYPE: shift,
    ACTION_GENERATE_TYPE: generate,
}

class ActionArgs:

    def __init__(self, embeddings, functions, stack, log_probs, outputs, tokens_embedding, batch_index, tokens_length, action):
        """
        :type embeddings: ActionEmbeddings
        :type functions: ActionFunctions
        :type stack: app.stacks.stack.Stack
        :type log_probs: ActionLogProbs
        :type outputs: ActionOutputs
        :type tokens_embedding: torch.Tensor
        :type batch_index: int
        :type tokens_length: int
        :type action: app.data.actions.action.Action
        """
        self.embeddings = embeddings
        self.functions = functions
        self.stack = stack
        self.outputs = outputs
        self.log_probs = log_probs
        self.tokens_embedding = tokens_embedding
        self.batch_index = batch_index
        self.tokens_length = tokens_length
        self.action = action

class ActionEmbeddings:

    def __init__(self, non_terminal, non_terminal_compose):
        """
        :type non_terminal: torch.nn.Embedding
        :type non_terminal_compose: torch.nn.Embedding
        """
        self.non_terminal = non_terminal
        self.non_terminal_compose = non_terminal_compose

class ActionFunctions:

    def __init__(self, representation2non_terminal_logits, logits2log_prob, composer, token_distribution):
        """
        :type representation2non_terminal_logits: torch.nn.Linear,
        :type logits2log_prob: torch.nn.LogSoftmax
        :type composer: app.composers.composer.Composer
        :type token_distribution: app.distributions.distribution.Distribution
        """
        self.representation2non_terminal_logits = representation2non_terminal_logits
        self.logits2log_prob = logits2log_prob
        self.composer = composer
        self.token_distribution = token_distribution

class ActionLogProbs:

    def __init__(self, representation, log_prob_base, action2index):
        """
        :type representation: torch.Tensor
        :type log_prob_base: torch.Tensor
        :type action2index: dict
        """
        self.representation = representation
        self.log_prob_base = log_prob_base
        self.action2index = action2index

class ActionOutputs:

    def __init__(self, open_non_terminals_count, token_index, token_counter):
        """
        :type open_non_terminals_count: int
        :type token_index: int
        :type token_counter: int
        """
        self.open_non_terminals_count = open_non_terminals_count
        self.token_index = token_index
        self.token_counter = token_counter
