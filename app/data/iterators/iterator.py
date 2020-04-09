import random

class Iterator:
    """
    An iterator is responsible for:
    * shuffling data
    * providing interface to iterate over data as batches of tensors
    """

    def __iter__(self):
        raise NotImplementedError('must be implemented by subclass')

    def size(self):
        """
        :rtype: int
        """
        raise NotImplementedError('must be implemented by subclass')

    def _shuffle_lists(self, *lists):
        zipped = list(zip(*lists))
        random.shuffle(zipped)
        shuffled_tuples = zip(*zipped)
        shuffled_lists = tuple(map(list, shuffled_tuples))
        return shuffled_lists
