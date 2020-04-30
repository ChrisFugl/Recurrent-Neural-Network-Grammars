from app.tasks.task import Task
import json
import logging
import os
import time
import torch

OUTPUT_FILENAME = 'samples.json'

class SampleTask(Task):

    def __init__(self, device, data, sampler):
        """
        :type device: torch.device
        :type data: str
        :type sampler: app.samplers.sampler.Sampler
        """
        super().__init__()
        self.logger = logging.getLogger('evaluate')
        self.device = device
        self.data = data
        self.sampler = sampler

    @torch.no_grad()
    def run(self):
        time_start = time.time()
        self.logger.info('Starting sampling')
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Data: {self.data}')
        self.logger.info(f'Sampler: {self.sampler}')
        samples = self.sampler.get_samples()
        time_stop = time.time()
        self.logger.info('Finished sampling')
        self.logger.info(f'Time taken: {time_stop - time_start:0.2f} s')
        self.logger.info('Saving samples to a file')
        save_path = self.save_samples(samples)
        self.logger.info(f'Saved samples at {save_path}')

    def save_samples(self, samples):
        output = self.samples2json(samples)
        path = self.get_path(OUTPUT_FILENAME)
        with open(path, 'w') as file:
            json.dump(output, file)
        return path

    def samples2json(self, samples):
        trees = []
        for sample in samples:
            gold = self.gold2json(sample.gold)
            predictions = list(map(self.prediction2json, sample.predictions))
            tree = {'gold': gold, 'predictions': predictions}
            trees.append(tree)
        output = {'trees': trees}
        return output

    def gold2json(self, gold):
        return {
            'actions': list(map(str, gold.actions)),
            'log_likelihood': gold.log_likelihood,
            'tokens': gold.tokens,
            'unknownified_tokens': gold.unknownified_tokens,
            'tags': gold.tags,
        }

    def prediction2json(self, prediction):
        return {
            'actions': list(map(str, prediction.actions)),
            'log_likelihood': prediction.log_likelihood,
        }

    def get_path(self, filename):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, filename)
        return path
