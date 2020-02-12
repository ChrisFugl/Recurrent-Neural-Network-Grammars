class Iterator:
    """
    An iterator is responsible for:
    * shuffling data
    * providing interface to iterate over data as batches of tensors
    """

    def __iter__(self):
        raise NotImplementedError('must be implemented by subclass')
