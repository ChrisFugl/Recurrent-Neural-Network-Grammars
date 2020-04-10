from app.samplers.sample import Sample
from app.samplers.sampler import Sampler
from torch.distributions import Categorical

class AncestralSampler(Sampler):

    def __init__(self, device, model, iterator, action_converter, posterior_scaling, samples, log=True):
        """
        :type device: torch.device
        :type model: torch.model
        :type iterator: app.data.iterators.iterator.Iterator
        :type action_converter: app.data.converters.action.ActionConverter
        :type posterior_scaling: float
        :type samples: int
        :type log: bool
        """
        super().__init__(device, action_converter, False, log=log)
        self.model = model
        self.iterator = iterator
        self.posterior_scaling = posterior_scaling
        self.samples = samples

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of app.samplers.sample.Sample
        """
        self.model.eval()
        best_actions = [None] * batch.size
        best_probs = [None] * batch.size
        best_log_prob = [None] * batch.size
        for _ in range(self.samples):
            pred_batch = self.sample(batch)
            pred_log_probs = self.model.batch_log_likelihood(pred_batch)
            pred_log_prob, pred_probs = self.batch_stats(pred_log_probs, pred_batch.actions.tensor, pred_batch.actions.lengths)
            for i in range(batch.size):
                if best_log_prob[i] is None or best_log_prob[i] < pred_log_prob[i]:
                    best_actions[i] = pred_batch.actions.actions[i]
                    best_probs[i] = pred_probs[i]
                    best_log_prob[i] = pred_log_prob[i]
        gold_log_probs = self.model.batch_log_likelihood(batch)
        gold_log_prob, gold_probs = self.batch_stats(gold_log_probs, batch.actions.tensor, batch.actions.lengths)
        samples = []
        for i in range(batch.size):
            g_actions = batch.actions.actions[i]
            g_tokens = batch.tokens.tokens[i]
            g_tags = batch.tags.tags[i]
            g_log_prob = gold_log_prob[i]
            g_probs = gold_probs[i]
            p_actions = best_actions[i]
            p_log_prob = best_log_prob[i]
            p_probs = best_probs[i]
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

    def get_valid_actions(self, state):
        return self.model.valid_actions(state)

    def sample_action(self, log_probs):
        """
        :type log_probs: torch.Tensor
        :rtype: int
        """
        distribution = Categorical(logits=log_probs)
        return distribution.sample()

    def __str__(self):
        return f'Ancestral(posterior_scaling={self.posterior_scaling}, samples={self.samples})'
