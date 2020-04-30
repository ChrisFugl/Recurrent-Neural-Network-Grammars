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

    def sample_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of app.samplers.sample.Sample
        """
        self.model.eval()
        pred_batch = self.sample(batch)
        pred_log_probs = self.model.batch_log_likelihood(pred_batch)
        pred_selected_log_probs, pred_log_likelihood = self.batch_log_probs(pred_log_probs, pred_batch.actions.tensor, pred_batch.actions.lengths)
        gold_log_probs = self.model.batch_log_likelihood(batch)
        gold_selected_log_probs, gold_log_likelihood = self.batch_log_probs(gold_log_probs, batch.actions.tensor, batch.actions.lengths)
        samples = []
        for i in range(batch.size):
            gold = (
                batch.actions.actions[i],
                batch.tokens.tokens[i],
                batch.unknownified_tokens.tokens[i],
                batch.tags.tags[i],
                gold_selected_log_probs[i],
                gold_log_likelihood[i]
            )
            prediction = (pred_batch.actions.actions[i], pred_selected_log_probs[i], pred_log_likelihood[i])
            predictions = [prediction]
            sample = Sample(gold, predictions)
            samples.append(sample)
        return samples

    def get_iterator(self):
        return self.iterator, self.iterator.size()

    def get_initial_state(self, batch):
        return self.model.initial_state(
            batch.tokens.tensor,
            batch.unknownified_tokens.tensor,
            batch.singletons,
            batch.tags.tensor,
            batch.tokens.lengths
        )

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
        return log_probs.argmax().cpu().item()

    def __str__(self):
        return f'Greedy(posterior_scaling={self.posterior_scaling})'
