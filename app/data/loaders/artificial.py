from app.data.loaders.loader import Loader
import hydra

class ArtificialLoader(Loader):

    def __init__(self, train_path, val_path, test_path):
        """
        :type path: str
        """
        self.train_path = hydra.utils.to_absolute_path(train_path)
        self.val_path = hydra.utils.to_absolute_path(val_path)
        self.test_path = hydra.utils.to_absolute_path(test_path)

    def load_train(self):
        # TODO
        pass

    def load_val(self):
        # TODO
        pass

    def load_test(self):
        # TODO
        pass
        
