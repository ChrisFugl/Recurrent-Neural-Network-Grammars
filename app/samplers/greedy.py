from app.samplers.sample import Sample
from app.samplers.sampler import Sampler

class GreedySampler(Sampler):

    def __init__(self, device, model, iterator, action_converter, posterior_scaling, log=True):
        """
        :type device: torch.device
        :type model: torch.model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type posterior_scaling: float
        :type log: bool
        """
        super().__init__(device, action_converter, log=log)
        self.model = model
        self.iterator = iterator
        self.posterior_scaling = posterior_scaling

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of app.samplers.sample.Sample
        """
        self.model.eval()
        predicted_batch = self.sample(batch)
        predicted_log_probs = self.model.batch_log_likelihood(predicted_batch)
        predicted_log_prob, predicted_probs = self.batch_stats(predicted_log_probs, predicted_batch.actions.lengths)
        gold_log_probs = self.model.batch_log_likelihood(batch)
        gold_log_prob, gold_probs = self.batch_stats(gold_log_probs, batch.actions.lengths)
        samples = []
        for i in range(batch.size):
            g_actions = batch.actions.actions[i]
            g_tokens = batch.tokens.tokens[i]
            g_tags = batch.tags.tags[i]
            g_log_prob = gold_log_prob[i]
            g_probs = gold_probs[i]
            p_actions = predicted_batch.actions.actions[i]
            p_log_prob = predicted_log_prob[i]
            p_probs = predicted_probs[i]
            sample = Sample(g_actions, g_tokens, g_tags, g_log_prob, g_probs, p_actions, p_log_prob, p_probs, None)
            samples.append(sample)
        return samples

    def get_iterator(self):
        return self.iterator, self.iterator.size()

    def get_initial_state(self, batch):
        return self.model.initial_state(batch.tokens.tensor, batch.tags.tensor, batch.tokens.lengths)

    def get_next_log_probs(self, state):
        return self.model.next_action_log_probs(state, posterior_scaling=self.posterior_scaling)

    def get_next_state(self, state, actions):
        return self.model.next_state(state, actions)

    def sample_actions(self, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: torch.Tensor
        """
        return log_probs.argmax(dim=1)

    def __str__(self):
        return f'Greedy(posterior_scaling={self.posterior_scaling})'
