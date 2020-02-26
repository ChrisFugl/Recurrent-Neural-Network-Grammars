import random

class Iterator:
    """
    An iterator is responsible for:
    * shuffling data
    * providing interface to iterate over data as batches of tensors
    """

    def __iter__(self):
        raise NotImplementedError('must be implemented by subclass')

    def _convert(self, converter, sequences):
        converted_sequences = []
        for sequence in sequences:
            converted_sequence = list(map(converter, sequence))
            converted_sequences.append(converted_sequence)
        return converted_sequences

    def _shuffle_lists(self, *lists):
        zipped = list(zip(*lists))
        random.shuffle(zipped)
        shuffled_tuples = zip(*zipped)
        shuffled_lists = tuple(map(list, shuffled_tuples))
        return shuffled_lists
