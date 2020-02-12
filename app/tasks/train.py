from app.tasks.task import Task

class TrainTask(Task):

    def __init__(self, train_iterator, val_iterator, model, optimizer, token_converter, action_converter):
        """
        :type train_iterator: app.data.iterators.iterator.Iterator
        :type val_iterator: app.data.iterators.iterator.Iterator
        :type model: app.models.model.Model
        :type optimizer: torch.optim.Optimizer
        :type token_converter: app.data.converters.token.TokenConverter
        :type action_converter: app.data.converters.action.ActionConverter
        """
        super().__init__()
        self._train_iterator = train_iterator
        self._val_iterator = val_iterator
        self._model = model
        self._optimizer = optimizer
        self._token_converter = token_converter
        self._action_converter = action_converter

    def run(self):
        # TODO
        print('TRAIN')
