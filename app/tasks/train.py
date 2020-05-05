from app.constants import ACTION_REDUCE_TYPE
from app.scores import scores_from_samples
from app.tasks.task import Task
from app.visualizations.actions import visualize_action_probs
from app.visualizations.gradients import visualize_gradients
from app.visualizations.trees import visualize_tree
import hydra
import logging
from math import exp
import numpy as np
from operator import itemgetter
import os
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainTask(Task):

    def __init__(self,
        device,
        iterator_train, iterator_val,
        model, loss, optimizer, learning_rate_scheduler,
        stopping_criterion, checkpoint, evaluator, sampler,
        log_train_every, load_checkpoint, token_count, tag_count, non_terminal_count, action_count
    ):
        """
        :type device: torch.device
        :type iterator_train: app.data.iterators.iterator.Iterator
        :type iterator_val: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type loss: app.losses.loss.Loss
        :type learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler
        :type optimizer: torch.optim.Optimizer
        :type stopping_criterion: app.stopping_criteria.stopping_criterion.StoppingCriterion
        :type checkpoint: app.checkpoints.checkpoint.Checkpoint
        :type evaluator: app.evaluators.evaluator.Evaluator
        :type sampler: app.samplers.sampler.Sampler
        :type log_train_every: int
        :type load_checkpoint: str
        :type token_count: int
        :type tag_count: int
        :type non_terminal_count: int
        :type action_count: int
        """
        super().__init__()
        self.device = device
        self.iterator_train = iterator_train
        self.iterator_val = iterator_val
        self.model = model
        self.loss = loss
        self.learning_rate_scheduler = learning_rate_scheduler
        self.optimizer = optimizer
        self.stopping_criterion = stopping_criterion
        self.checkpoint = checkpoint
        self.evaluator = evaluator
        self.sampler = sampler
        self.log_train_every = log_train_every
        self.token_count = token_count
        self.tag_count = tag_count
        self.non_terminal_count = non_terminal_count
        self.action_count = action_count
        working_directory = os.getcwd()
        tensorboard_directory = os.path.join(working_directory, 'tb')
        self.writer = SummaryWriter(log_dir=tensorboard_directory)
        self.total_time_offset = 0
        self.start_epoch = 0
        self.start_batch_count = 0
        self.start_sequence_count = 0
        self.logger = logging.getLogger('train')
        self.uses_gpu = self.device.type == 'cuda'
        self.gradient_batch = self.get_gradient_batch(iterator_val)
        if load_checkpoint is not None:
            self.load_checkpoint(load_checkpoint)

    def run(self):
        self.start_timestamp = time.time()
        self.logger.info('Starting training')
        self.logger.info(f'Saving output to {os.getcwd()}')
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Tokens: {self.token_count:,}')
        self.logger.info(f'Part-of-speech tags: {self.tag_count:,}')
        self.logger.info(f'Non-terminals: {self.non_terminal_count:,}')
        self.logger.info(f'Actions: {self.action_count:,}')
        self.logger.info(f'Parameters: {self.count_parameters():,}')
        self.logger.info(f'Model:\n{self.model}')
        batch_count = self.start_batch_count
        sequence_count = self.start_sequence_count
        epoch = self.start_epoch
        last_log_count = self.start_sequence_count
        log_values = []
        best_val_score = None
        self.model.train()
        if self.evaluator.should_evaluate(epoch, batch_count, pretraining=True):
            best_val_score = self.evaluate(epoch, batch_count, sequence_count, best_val_score)
        # assumes a single learning rate for all parameters
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.log_epoch(epoch, batch_count, sequence_count, learning_rate)
        while True:
            train_epoch_outputs = self.train_epoch(epoch, batch_count, sequence_count, last_log_count, log_values, best_val_score)
            epoch, batch_count, sequence_count, last_log_count, done, log_values, best_val_score = train_epoch_outputs
            if done:
                break
            if self.evaluator.should_evaluate(epoch, batch_count, end_of_epoch=True):
                best_val_score = self.evaluate(epoch, batch_count, sequence_count, best_val_score)
            self.stopping_criterion.add_epoch(epoch)
            if self.stopping_criterion.is_done():
                break
        self.evaluate(epoch, batch_count, sequence_count, best_val_score)
        time_stop = time.time()
        time_seconds = time_stop - self.start_timestamp
        time_hours = time_seconds / 3600
        time_days = time_hours / 24
        self.logger.info(f'Training time: {time_seconds:0.2f} s/{time_hours:0.2f} h/{time_days:0.2f} d')
        self.logger.info('Finished training')
        self.writer.close()

    def train_epoch(self, epoch, batch_count, sequence_count, last_log_count, log_values, best_val_score):
        time_epoch_start = time.time()
        start_of_epoch = True
        for batch in self.iterator_train:
            batch_count += 1
            sequence_count += batch.size
            batch_outputs = self.train_batch(batch)
            log_values.append(batch_outputs)
            if self.log_train_every <= sequence_count - last_log_count:
                self.log_train(log_values, sequence_count)
                last_log_count = sequence_count
                log_values = []
            if self.evaluator.should_evaluate(epoch, batch_count):
                best_val_score = self.evaluate(epoch, batch_count, sequence_count, best_val_score)
            if self.stopping_criterion.is_done():
                return epoch, batch_count, sequence_count, last_log_count, True, log_values, best_val_score
            if self.checkpoint.should_save_checkpoint(epoch, batch_count, start_of_epoch):
                self.save_checkpoint(epoch, batch_count, sequence_count)
            start_of_epoch = False
        self.learning_rate_scheduler.step()
        # assumes a single learning rate for all parameters
        learning_rate = self.learning_rate_scheduler.get_last_lr()[0]
        epoch += 1
        time_epoch_stop = time.time()
        time_epoch = time_epoch_stop - time_epoch_start
        time_total = self.get_time_elapsed()
        self.log_epoch(epoch, batch_count, sequence_count, learning_rate)
        self.writer.add_scalar('time/epoch_s', time_epoch, sequence_count)
        self.writer.add_scalar('time/total_s', time_total, sequence_count)
        return epoch, batch_count, sequence_count, last_log_count, False, log_values, best_val_score

    def train_batch(self, batch):
        self.start_measure_memory()
        time_batch_start = time.time()
        batch_log_probs = self.model.batch_log_likelihood(batch)
        time_batch_stop = time.time()
        loss = self.loss(batch_log_probs, batch.actions.tensor, batch.actions.lengths)
        time_optimize_start = time.time()
        self.optimize(loss)
        loss_scalar = loss.cpu().item()
        time_optimize_stop = time.time()
        time_batch = time_batch_stop - time_batch_start
        time_optimize = time_optimize_stop - time_optimize_start
        actions_per_second = self.count_actions(batch) / time_batch
        tokens_per_second = self.count_tokens(batch) / time_batch
        sentences_per_second = batch.size / time_batch
        memory = self.stop_measure_memory()
        if memory is None:
            allocated_gb, reserved_gb = None, None
        else:
            allocated_gb, reserved_gb = memory
        return loss_scalar, allocated_gb, reserved_gb, actions_per_second, tokens_per_second, sentences_per_second, time_batch, time_optimize

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, epoch, batch_count, sequence_count, best_score):
        self.start_measure_memory()
        time_evaluate_start = time.time()
        self.model.eval()
        loss_train = self.evaluate_train_loss()
        loss_val, total_actions, total_tokens, total_sentences, time_val = self.evaluate_val_loss()
        loss_diff = loss_val - loss_train
        if self.sampler is not None:
            samples, scores, time_sample = self.sample()
            f1, precision, recall = scores
        self.model.train()
        gradients_all = self.make_gradient_visualizations()
        self.model.eval()
        if self.sampler is not None:
            sample_index = random.randint(0, len(samples) - 1)
            sample = samples[sample_index]
            groundtruth_tree, predicted_tree = self.make_tree_visualizations(sample)
            groundtruth_probs, predicted_probs = self.make_action_visualizations(sample)
        time_evaluate_stop = time.time()
        time_evaluate = time_evaluate_stop - time_evaluate_start
        actions_per_second = total_actions / time_val
        tokens_per_second = total_tokens / time_val
        sentences_per_second = total_sentences / time_val
        self.writer.add_scalar('evaluation/loss_train', loss_train, sequence_count)
        self.writer.add_scalar('evaluation/loss_val', loss_val, sequence_count)
        self.writer.add_scalar('evaluation/loss_val - loss_train', loss_diff, sequence_count)
        self.stopping_criterion.add_val_loss(loss_val)
        memory = self.stop_measure_memory()
        if memory is not None:
            allocated_gb, reserved_gb = memory
            self.writer.add_scalar('memory/allocated_gb_val', allocated_gb, sequence_count)
            self.writer.add_scalar('memory/reserved_gb_val', reserved_gb, sequence_count)
        self.writer.add_figure('gradients/all', gradients_all, sequence_count)
        self.writer.add_scalar('time/evaluate_s', time_evaluate, sequence_count)
        self.writer.add_scalar('time/val_s', time_val, sequence_count)
        self.writer.add_scalar('time/actions_per_s_val', actions_per_second, sequence_count)
        self.writer.add_scalar('time/tokens_per_s_val', tokens_per_second, sequence_count)
        self.writer.add_scalar('time/sentences_per_s_val', sentences_per_second, sequence_count)
        if self.sampler is not None:
            self.writer.add_figure('tree/groundtruth', groundtruth_tree, sequence_count)
            self.writer.add_figure('tree/predicted', predicted_tree, sequence_count)
            self.writer.add_figure('actions/groundtruth', groundtruth_probs, sequence_count)
            self.writer.add_figure('actions/predicted', predicted_probs, sequence_count)
            self.writer.add_scalar('time/sample_s', time_sample, sequence_count)
            self.writer.add_scalar('evaluation/f1', f1, sequence_count)
            self.writer.add_scalar('evaluation/precision', precision, sequence_count)
            self.writer.add_scalar('evaluation/recall', recall, sequence_count)
            self.logger.info(f'epoch={epoch}, batch={batch_count}, sequences={sequence_count},  loss_train={loss_train:0.6f}, loss_val={loss_val:0.6f}, f1_val={f1:0.2f}')
            score = f1
            is_new_best_score = best_score is None or best_score < score
        else:
            self.logger.info(f'epoch={epoch}, batch={batch_count}, sequences={sequence_count},  loss_train={loss_train:0.6f}, loss_val={loss_val:0.6f}')
            score = loss_val
            is_new_best_score = best_score is None or score < best_score
        self.evaluator.evaluation_finished()
        self.model.train()
        # save model with best performing score
        if is_new_best_score:
            if best_score is not None:
                self.save()
            return score
        else:
            return best_score

    @torch.no_grad()
    def evaluate_train_loss(self):
        """
        Evaluate train loss against the (approximate) same number of
        sentences as in the validationset.
        """
        total_val_sentences = self.iterator_val.size()
        sentences = 0
        losses = []
        for batch in self.iterator_train:
            log_probs = self.model.batch_log_likelihood(batch)
            loss = self.loss(log_probs, batch.actions.tensor, batch.actions.lengths)
            loss_scalar = loss.cpu().item()
            losses.append(loss_scalar)
            sentences += batch.size
            if total_val_sentences <= sentences:
                break
        loss = sum(losses) / len(losses)
        return loss

    @torch.no_grad()
    def evaluate_val_loss(self):
        losses = []
        time_taken = 0
        total_actions = 0
        total_tokens = 0
        total_sentences = self.iterator_val.size()
        for batch in self.iterator_val:
            time_start = time.time()
            log_probs = self.model.batch_log_likelihood(batch)
            time_stop = time.time()
            time_taken += time_stop - time_start
            loss = self.loss(log_probs, batch.actions.tensor, batch.actions.lengths)
            loss_scalar = loss.cpu().item()
            losses.append(loss_scalar)
            total_actions += self.count_actions(batch)
            total_tokens += self.count_tokens(batch)
        loss = sum(losses) / len(losses)
        return loss, total_actions, total_tokens, total_sentences, time_taken

    @torch.enable_grad()
    def make_gradient_visualizations(self):
        log_probs = self.model.batch_log_likelihood(self.gradient_batch)
        loss = self.loss(log_probs, self.gradient_batch.actions.tensor, self.gradient_batch.actions.lengths)
        self.model.zero_grad()
        loss.backward()
        gradients_all = visualize_gradients(self.model.named_parameters())
        return gradients_all

    def make_tree_visualizations(self, sample):
        groundtruth = visualize_tree(sample.gold.actions, sample.gold.tokens)
        # assumes that only a single tree is predicted
        predicted = visualize_tree(sample.predictions[0].actions, sample.gold.tokens)
        return groundtruth, predicted

    def make_action_visualizations(self, sample):
        gold_probs = [exp(prob) for prob in sample.gold.log_probs]
        groundtruth = visualize_action_probs(sample.gold.actions, gold_probs)
        # assumes that only a single tree is predicted
        prediction = sample.predictions[0]
        pred_probs = [exp(prob) for prob in prediction.log_probs]
        predicted = visualize_action_probs(prediction.actions, pred_probs)
        return groundtruth, predicted

    @torch.no_grad()
    def sample(self):
        time_start = time.time()
        samples = self.sampler.get_samples()
        time_stop = time.time()
        time_taken = time_stop - time_start
        tokens = [sample.gold.tokens for sample in samples]
        tags = [sample.gold.tags for sample in samples]
        gold_trees = [sample.gold.actions for sample in samples]
        predicted_trees = [sample.predictions[0].actions for sample in samples] # assumes that only one tree is sampled
        scores, _, _, _ = scores_from_samples(tokens, tags, gold_trees, predicted_trees)
        return samples, scores, time_taken

    def save(self):
        file_name = 'model.pt'
        directory = hydra.utils.to_absolute_path(os.getcwd())
        file_path = os.path.join(directory, file_name)
        self.logger.info(f'Saving model at {file_path}')
        self.model.save(file_path)

    def save_checkpoint(self, epoch, batch_count, sequence_count):
        time_checkpoint_start = time.time()
        checkpoints_directory = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(checkpoints_directory, exist_ok=True)
        checkpoint_name = f'epoch_{epoch}_batch_{batch_count}_seq_{sequence_count}.pt'
        checkpoint_path = os.path.join(checkpoints_directory, checkpoint_name)
        self.logger.info(f'Saving checkpoint at {checkpoint_path}')
        checkpoint = {
            'batch_count': batch_count,
            'sequence_count': sequence_count,
            'epoch': epoch,
            'learning_rate_scheduler': self.learning_rate_scheduler.state_dict(),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stopping_criterion': self.stopping_criterion.state_dict(),
            'time_elapsed': time.time() - self.start_timestamp,
        }
        torch.save(checkpoint, checkpoint_path)
        time_checkpoint_stop = time.time()
        time_checkpoint = time_checkpoint_stop - time_checkpoint_start
        self.writer.add_scalar('time/checkpoint_s', time_checkpoint, sequence_count)

    def load_checkpoint(self, path):
        absolute_path = hydra.utils.to_absolute_path(path)
        self.logger.info(f'Loading checkpoint from {absolute_path}')
        checkpoint = torch.load(absolute_path)
        self.start_batch_count = checkpoint['batch_count']
        self.start_sequence_count = checkpoint['sequence_count']
        self.start_epoch = checkpoint['epoch']
        self.total_time_offset = checkpoint['time_elapsed']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['learning_rate_scheduler'])
        self.stopping_criterion.load_state_dict(checkpoint['stopping_criterion'])

    def log_epoch(self, epoch, batch_count, sequence_count, learning_rate):
        self.logger.info(f'epoch={epoch}, batch={batch_count}, sequences={sequence_count}, learning rate={learning_rate}')
        self.writer.add_scalar('training/epoch', epoch, sequence_count)
        self.writer.add_scalar('training/learning_rate', learning_rate, sequence_count)

    def count_parameters(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        count = sum([np.prod(parameter.size()) for parameter in parameters])
        return count

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

    def count_actions(self, batch):
        return sum(map(len, batch.actions.actions))

    def count_tokens(self, batch):
        return sum(map(len, batch.tokens.tokens))

    def get_time_elapsed(self):
        return self.total_time_offset + (time.time() - self.start_timestamp)

    def log_train(self, values, sequence_count):
        # return loss_scalar, memory, actions_per_second, tokens_per_second, sentences_per_second, time_batch, time_optimize
        loss = self.mean_from_tuple_index(values, 0)
        allocated_gb = self.mean_from_tuple_index(values, 1)
        reserved_gb = self.mean_from_tuple_index(values, 2)
        actions_per_second = self.mean_from_tuple_index(values, 3)
        tokens_per_second = self.mean_from_tuple_index(values, 4)
        sentences_per_second = self.mean_from_tuple_index(values, 5)
        time_batch = self.mean_from_tuple_index(values, 6)
        time_optimize = self.mean_from_tuple_index(values, 7)
        self.writer.add_scalar('training/loss_train', loss, sequence_count)
        self.writer.add_scalar('time/actions_per_s_train', actions_per_second, sequence_count)
        self.writer.add_scalar('time/tokens_per_s_train', tokens_per_second, sequence_count)
        self.writer.add_scalar('time/sentences_per_s_train', sentences_per_second, sequence_count)
        self.writer.add_scalar('time/batch_s', time_batch, sequence_count)
        self.writer.add_scalar('time/optimize_s', time_optimize, sequence_count)
        if allocated_gb is not None:
            self.writer.add_scalar('memory/allocated_gb_train', allocated_gb, sequence_count)
        if reserved_gb is not None:
            self.writer.add_scalar('memory/reserved_gb_train', reserved_gb, sequence_count)
        return loss

    def mean_from_tuple_index(self, values, index):
        if values[0][index] is None:
            return None
        elements = list(map(itemgetter(index), values))
        return sum(elements) / len(elements)

    def get_gradient_batch(self, iterator):
        """
        Find first batch to have at least two REDUCE actions.
        This is needed as the gradient visualization may be
        incorrect for the composition parameters, when it is
        evaluated against a batch that only has a single
        REDUCE action at the end of the action sequence.
        """
        for batch in iterator:
            for actions in batch.actions.actions:
                reduce_count = 0
                for action in actions:
                    if action.type() == ACTION_REDUCE_TYPE:
                        reduce_count += 1
                        if 1 < reduce_count:
                            return batch
        raise Exception('Could not find a batch with at least two REDUCE actions.')
