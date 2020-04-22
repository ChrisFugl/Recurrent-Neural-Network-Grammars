from app.tasks.task import Task
import logging
from math import ceil
import os
import pandas as pd
import time
import torch

RESULTS_FILENAME = 'time_stats.csv'
RESULTS_HEADERS = ['sentences', 'tokens', 'actions', 'memory_allocated_gb', 'memory_reserved_gb', 'time_s']

class TimeStatsTask(Task):

    def __init__(self, device, model, iterator):
        """
        :type device: torch.device
        :type model: app.models.model.Model
        :type iterator: app.data.iterators.iterator.Iterator
        """
        super().__init__()
        self.model = model
        self.iterator = iterator
        self.logger = logging.getLogger('time_stats')
        self.device = device
        self.uses_gpu = self.device.type == 'cuda'

    @torch.no_grad()
    def run(self):
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Batch size: {self.iterator.get_batch_size()}')
        self.logger.info(f'Model:\n{self.model}')
        self.logger.info('Started measuring')
        time_start = time.time()
        timings = self.measure_timings()
        time_stop = time.time()
        self.logger.info('Finished measuring')
        self.logger.info(f'Time taken: {time_stop - time_start:0.2} s')
        self.log_timings(timings)
        self.save_timings(timings)

    def measure_timings(self):
        results = []
        counter = 0
        batch_size = self.iterator.get_batch_size()
        data_size = self.iterator.size()
        batch_total = ceil(data_size / batch_size)
        log_threshold = batch_total / 10 # log progress approximately every 10th batch
        self.model.eval()
        for i, batch in enumerate(self.iterator):
            self.start_measure_memory()
            time_start = time.time()
            self.model.batch_log_likelihood(batch)
            time_stop = time.time()
            memory = self.stop_measure_memory()
            if memory is None:
                allocated_gb, reserved_gb = None, None
            else:
                allocated_gb, reserved_gb = memory
            sentences = batch.size
            tokens = self.count_tokens(batch)
            actions = self.count_actions(batch)
            time_s = time_stop - time_start
            result = [sentences, tokens, actions, allocated_gb, reserved_gb, time_s]
            results.append(result)
            if log_threshold <= counter:
                self.logger.info(f'Batch {i + 1} / {batch_total} ({(i + 1)/batch_total:0.2%})')
                counter = -1
            counter += 1
        dataframe = pd.DataFrame(results, columns=RESULTS_HEADERS)
        return dataframe

    def count_actions(self, batch):
        return sum(map(len, batch.actions.actions))

    def count_tokens(self, batch):
        return sum(map(len, batch.tokens.tokens))

    def log_timings(self, timings):
        self.logger.info(f'time/batch (mean) = {timings.time_s.mean():0.2f} s')
        self.logger.info(f'time/batch (std)  = {timings.time_s.std():0.2f} s')
        sentences_per_s = timings.sentences / timings.time_s
        self.logger.info(f'sentences/s (mean) = {sentences_per_s.mean():0.2f}')
        self.logger.info(f'sentences/s (std)  = {sentences_per_s.std():0.2f}')
        tokens_per_s = timings.tokens / timings.time_s
        self.logger.info(f'tokens/s (mean)    = {tokens_per_s.mean():0.2f}')
        self.logger.info(f'tokens/s (std)     = {tokens_per_s.std():0.2f}')
        actions_per_s = timings.actions / timings.time_s
        self.logger.info(f'actions/s (mean)   = {actions_per_s.mean():0.2f}')
        self.logger.info(f'actions/s (std)    = {actions_per_s.std():0.2f}')

    def save_timings(self, timings):
        path = os.path.join(os.getcwd(), RESULTS_FILENAME)
        self.logger.info(f'Saving output to {path}')
        timings.to_csv(path, index=False)

    def byte2gb(self, byte_amount):
        return byte_amount / (1024 ** 3)

    def start_measure_memory(self):
        if self.uses_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

    def stop_measure_memory(self):
        if self.uses_gpu:
            allocated_gb = self.byte2gb(torch.cuda.max_memory_allocated(self.device))
            reserved_gb = self.byte2gb(torch.cuda.max_memory_reserved(self.device))
            return allocated_gb, reserved_gb
        return None
