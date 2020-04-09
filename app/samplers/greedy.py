from app.data.batch import Batch
from app.data.batch_utils import sequences2lengths, sequences2tensor
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
        self.device = device
        self.model = model
        self.iterator = iterator
        self.action_converter = action_converter
        self.posterior_scaling = posterior_scaling

    def evaluate_batch(self, batch):
        """
        :type batch: app.data.batch.Batch
        :rtype: list of app.samplers.sample.Sample
        """
        self.model.eval()
        predicted_actions = self.sample(batch.tokens.tensor, batch.tags.tensor, batch.tokens.lengths)
        predicted_tensor = sequences2tensor(self.device, self.action_converter.action2integer, predicted_actions)
        predicted_lengths = sequences2lengths(self.device, predicted_actions)
        predicted_batch = Batch(
            predicted_tensor, predicted_lengths, predicted_actions,
            batch.tokens.tensor, batch.tokens.lengths, batch.tokens.tokens,
            batch.tags.tensor, batch.tags.lengths, batch.tags.tags,
        )
        predicted_log_probs = self.model.batch_log_likelihood(predicted_batch)
        predicted_log_prob, predicted_probs = self.stats(predicted_log_probs, predicted_lengths)
        gold_log_probs = self.model.batch_log_likelihood(batch)
        gold_log_prob, gold_probs = self.stats(gold_log_probs, batch.actions.lengths)
        samples = []
        for i in range(batch.size):
            sample = Sample(
                batch.actions.actions[i], batch.tokens.tokens[i], batch.tags.tags[i], gold_log_prob[i], gold_probs[i],
                predicted_actions[i], predicted_log_prob[i], predicted_probs[i],
                None
            )
            samples.append(sample)
        return samples

    def get_iterator(self):
        return self.iterator, self.iterator.size()

    def sample(self, tokens, tags, lengths):
        """
        :type tokens: torch.Tensor
        :type tags: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of list of app.samplers.sample.Sample
        """
        state = self.model.initial_state(tokens, tags, lengths)
        lengths_list = [length.cpu().item() for length in lengths]
        batch_size = lengths.size(0)
        batch_samples = [[] for _ in range(batch_size)]
        finished_sampling = [False] * batch_size
        while not all(finished_sampling):
            log_probs = self.model.next_action_log_probs(state, posterior_scaling=self.posterior_scaling)
            samples = log_probs.argmax(dim=1)
            actions = []
            for i, finished in enumerate(finished_sampling):
                if finished:
                    # None represents padding
                    actions.append(None)
                else:
                    sample = samples[i]
                    action = self.action_converter.integer2action(sample)
                    batch_samples[i].append(action)
                    finished_sampling[i] = self.is_finished_sampling(batch_samples[i], lengths_list[i])
                    actions.append(action)
            state = self.model.next_state(state, actions)
        return batch_samples

    def stats(self, batch_log_probs, lengths):
        """
        :param batch_log_probs: tensor, S x B x A
        :type batch_log_probs: torch.Tensor
        :type lengths: torch.Tensor
        :rtype: list of float, list of list of float
        """
        summed = batch_log_probs.sum(dim=2)
        log_probs = [self.get_log_prob(summed[:length, i]) for i, length in enumerate(lengths)]
        probs = [self.get_probs(summed[:length, i]) for i, length in enumerate(lengths)]
        return log_probs, probs

    def get_log_prob(self, log_probs):
        return log_probs.sum().cpu().item()

    def get_probs(self, log_probs):
        return [prob.cpu().item() for prob in log_probs.exp()]

    def __str__(self):
        return f'Greedy(posterior_scaling={self.posterior_scaling})'
