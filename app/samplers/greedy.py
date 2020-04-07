from app.samplers.sample import Sample
from app.samplers.sampler import Sampler

class GreedySampler(Sampler):

    def __init__(self, device, model, iterator, action_converter, posterior_scaling, log=True):
        """
        :type device: torch.device
        :type action_converter: app.data.converters.action.ActionConverter
        :type model: torch.model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type posterior_scaling: float
        :type log: bool
        """
        super().__init__(log=log)
        self._device = device
        self._model = model
        self._iterator = iterator
        self._action_converter = action_converter
        self._posterior_scaling = posterior_scaling

    def evaluate_element(self, batch, batch_index):
        """
        :type batch: app.data.batch.Batch
        :type batch_index: int
        :rtype: app.samplers.sample.Sample
        """
        self._model.eval()
        element = batch.get(batch_index)
        tokens_tensor = element.tokens.tensor[:element.tokens.length, :]
        tags_tensor = element.tags.tensor[:element.tags.length, :]
        predicted_tree = self._sample_from_tokens_tensor(tokens_tensor, tags_tensor)
        predicted_tree_tensor = self._actions2tensor(self._action_converter, predicted_tree)
        predicted_tree_log_probs = self._model.tree_log_probs(tokens_tensor, tags_tensor, predicted_tree_tensor, predicted_tree)
        predicted_tree_probs = [prob.cpu().item() for prob in predicted_tree_log_probs.sum(dim=1).exp()]
        predicted_tree_log_prob = predicted_tree_log_probs.sum().cpu().item()
        gold_tree = element.actions.actions
        gold_tree_tensor = element.actions.tensor[:element.actions.length, :]
        gold_log_probs = self._model.tree_log_probs(tokens_tensor, tags_tensor, gold_tree_tensor, gold_tree)
        gold_probs = [prob.cpu().item() for prob in gold_log_probs.sum(dim=1).exp()]
        gold_log_prob = gold_log_probs.sum().cpu().item()
        return Sample(
            gold_tree, element.tokens.tokens, element.tags.tags, gold_log_prob, gold_probs,
            predicted_tree, predicted_tree_log_prob, predicted_tree_probs,
            None
        )

    def get_batch_size(self, batch):
        """
        :rtype: int
        """
        return batch.size

    def get_iterator(self):
        return self._iterator, self._iterator.size()

    def _sample_from_tokens_tensor(self, tokens, tags):
        tokens_length = len(tokens)
        actions = []
        state = self._model.initial_state(tokens, tags)
        while not self._is_finished_sampling(actions, tokens_length):
            log_probs, index2action_index = self._model.next_action_log_probs(state, posterior_scaling=self._posterior_scaling)
            sample = index2action_index[log_probs.argmax()]
            action = self._action_converter.integer2action(sample)
            actions.append(action)
            state = self._model.next_state(state, action)
        return actions

    def __str__(self):
        return f'Greedy(posterior_scaling={self._posterior_scaling})'
