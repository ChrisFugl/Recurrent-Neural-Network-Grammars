class Loader:

    def load_train(self):
        """
        Load trainingset.
        """
        raise NotImplementedError('must be implemented by subclass')

    def load_val(self):
        """
        Load validationset.
        """
        raise NotImplementedError('must be implemented by subclass')

    def load_test(self):
        """
        Load testset.
        """
        raise NotImplementedError('must be implemented by subclass')
