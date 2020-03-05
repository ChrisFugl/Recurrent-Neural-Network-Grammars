from app.models.model import Model
from app.models.rnng.actions import call_action, ActionArgs, ActionOutputs, ActionFunctions, ActionEmbeddings, ActionStructures
from joblib import Parallel, delayed
import torch
from torch import nn

# base actions: (REDUCE, GENERATE, NON_TERMINAL) or (REDUCE, SHIFT, NON_TERMINAL)
ACTIONS_COUNT = 3

class RNNG(Model):

    def __init__(
        self, device, generative,
        action_embedding, token_embedding,
        non_terminal_embedding, non_terminal_compose_embedding,
        action_history, token_buffer, stack,
        representation, representation_size,
        composer,
        token_distribution,
        non_terminal_count,
        action_set,
        threads,
        reverse_tokens,
    ):
        """
        :type device: torch.device
        :type generative: bool
        :type action_embedding: torch.Embedding
        :type token_embedding: torch.Embedding
        :type non_terminal_embedding: torch.Embedding
        :type non_terminal_compose_embedding: torch.Embedding
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type stack: app.stacks.stack.Stack
        :type representation: app.representations.representation.Representation
        :type representation_size: int
        :type composer: app.composers.composer.Composer
        :type token_distribution: app.distributions.distribution.Distribution
        :type non_terminal_count: int
        :type action_set: app.data.action_set.ActionSet
        :type threads: int
        :type reverse_tokens: bool
        """
        super().__init__()
        self._device = device
        self._generative = generative
        self._threads = threads
        self._reverse_tokens = reverse_tokens
        self._action_set = action_set
        self._action_embedding = action_embedding
        self._non_terminal_embedding = non_terminal_embedding
        self._non_terminal_compose_embedding = non_terminal_compose_embedding
        self._token_embedding = token_embedding
        self._action_history = action_history
        self._token_buffer = token_buffer
        self._stack = stack
        self._representation = representation
        self._representation2logits = nn.Linear(in_features=representation_size, out_features=ACTIONS_COUNT, bias=True)
        self._composer = composer
        self._logits2log_prob = nn.LogSoftmax(dim=2)
        self._representation2non_terminal_logits = nn.Linear(in_features=representation_size, out_features=non_terminal_count, bias=True)
        if self._generative:
            self._token_distribution = token_distribution

        self._action_embeddings = ActionEmbeddings(self._non_terminal_embedding, self._non_terminal_compose_embedding)
        self._action_functions = ActionFunctions(self._representation2non_terminal_logits, self._logits2log_prob, composer, token_distribution)

        # TODO: initialize

    def log_likelihood(self, batch, posterior_scaling=1.0):
        """
        Compute log likelihood of each sentence/tree in a batch.

        :type batch: app.data.batch.Batch
        posterior_scaling=1.0
        :rtype: torch.Tensor
        """
        action_history = self._action_history.new()
        token_buffer = self._token_buffer.new()

        actions_embedding = self._action_embedding(batch.actions.tensor)
        tokens_embedding = self._token_embedding(batch.tokens.tensor)

        if self._reverse_tokens:
            tokens_embedding = tokens_embedding.flip([0])

        # add to action history and token buffer
        actions_after_first_embedding = actions_embedding[1:]
        action_history.add(actions_after_first_embedding)
        token_buffer.add(tokens_embedding)

        # stack operations
        jobs_args = []
        for batch_index in range(batch.size):
            element = batch.get(batch_index)
            job_args = (action_history, token_buffer, actions_embedding, tokens_embedding, element)
            jobs_args.append(job_args)
        if 1 < self._threads:
            get_log_probs = delayed(self._log_probs)
            log_probs_list = Parallel(n_jobs=self._threads, backend='threading')(get_log_probs(*job_args) for job_args in jobs_args)
        else:
            log_probs_list = list(map(lambda job_args: self._log_probs(*job_args), jobs_args))
        log_probs = torch.stack(log_probs_list, dim=1)

        return log_probs

    def initial_state(self, tokens):
        """
        Get initial state of model in a parse.

        :type tokens: torch.Tensor
        :returns: initial state
        :rtype: app.models.rnng.state.RNNGState
        """
        stack = self._stack.new()
        action_history = self._action_history.new()
        token_buffer = self._token_buffer.new()
        tokens_embedding = self._token_embedding(tokens)
        token_buffer.add(tokens_embedding)
        tokens_length = tokens.size(0)
        # TODO
        return None

    def next_state(self, previous_state, action):
        """
        Advance state of the model to the next state.

        :param previous_state: model specific previous state
        :type action: app.data.actions.action.Action
        """
        raise NotImplementedError('must be implemented by subclass')

    def next_action_log_probs(self, state, posterior_scaling=1.0):
        """
        Compute log probability of every action given the current state.

        :param state: state of a parse
        :rtype: torch.Tensor
        """
        raise NotImplementedError('must be implemented by subclass')

    def save(self, path):
        """
        Save model parameters.

        :type path: str
        """
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        """
        Load model parameters from file.

        :type path: str
        """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def _log_probs(self, action_history, token_buffer, actions_embedding, tokens_embedding, element):
        """
        :type action_history: app.memories.memory.Memory
        :type token_buffer: app.memories.memory.Memory
        :type actions_embedding: torch.nn.Embedding
        :type tokens_embedding: torch.nn.Embedding
        :type element: app.data.batch.BatchElement
        """
        stack = self._stack.new()
        action_structures = ActionStructures(stack, token_buffer)
        token_index = 0 if not self._reverse_tokens else element.tokens.max_length - 1
        token_counter = 0

        # first action is always NT(S), use this to initialize stack
        action_first = element.actions.actions[0]
        self._push_to_stack(stack, actions_embedding, 0, element.index, action_first)
        open_non_terminals_count = 1
        log_probs = torch.zeros((element.actions.max_length - 1,), dtype=torch.float, device=self._device, requires_grad=False)

        for action_counter in range(1, element.actions.length):
            action_index = action_counter - 1
            action = element.actions.actions[action_counter]
            representation = self._representation(action_history, stack, token_buffer, action_index, token_index, element.index)
            valid_actions, action2index = self._action_set.valid_actions(element.tokens.length, token_counter, stack, open_non_terminals_count)
            assert action.index() in action2index, f'{action} is not a valid action. (action2index = {action2index})'
            logits_base = self._representation2logits(representation)
            logits_base_valid = logits_base[:, :, valid_actions]
            log_prob_base_valid = self._logits2log_prob(logits_base_valid)

            # get log probability of action
            action_outputs = ActionOutputs(open_non_terminals_count, token_index, token_counter)
            action_args = ActionArgs(
                self._action_embeddings, self._action_functions, action_structures, action_outputs,
                tokens_embedding, representation, log_prob_base_valid, action2index, element, action
            )
            action_log_prob, open_non_terminals_count, token_index, token_counter = call_action(action.type(), action_args)
            log_probs[action_index] = action_log_prob

        return log_probs

    def _push_to_stack(self, stack, embeddings, item_index, batch_index, action):
        action_embedding = embeddings[item_index, batch_index].unsqueeze(dim=0).unsqueeze(dim=0)
        return stack.push(action_embedding, action)

    def __str__(self):
        return (
            'RNNG(\n'
            + f'  action_history={self._action_history}\n'
            + f'  token_buffer={self._token_buffer}\n'
            + f'  stack={self._stack}\n'
            + f'  representation={self._representation}\n'
            + f'  composer={self._composer}\n'
            + ('' if not self._generative else f'  token_distribution={self._token_distribution}\n')
            + ')'
        )

    # override train and eval methods to ensure that subcomponents are also put in train/eval mode
    def train(self, mode=True):
        super().train(mode=mode)
        self._action_history.train(mode=mode)
        self._token_buffer.train(mode=mode)
        self._stack.train(mode=mode)
        self._representation.train(mode=mode)
        self._composer.train(mode=mode)
        if self._generative:
            self._token_distribution.train(mode=mode)

    def eval(self):
        super().eval()
        self._action_history.eval()
        self._token_buffer.eval()
        self._stack.eval()
        self._representation.eval()
        self._composer.eval()
        if self._generative:
            self._token_distribution.eval()
