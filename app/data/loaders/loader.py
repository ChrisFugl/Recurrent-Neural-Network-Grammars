class Loader:
    """
    A loader is responsible for:
    * retrieving data from files
    * tokenization
    """

    def load_train(self):
        raise NotImplementedError('must be implemented by subclass')

    def load_val(self):
        raise NotImplementedError('must be implemented by subclass')

    def load_test(self):
        raise NotImplementedError('must be implemented by subclass')
