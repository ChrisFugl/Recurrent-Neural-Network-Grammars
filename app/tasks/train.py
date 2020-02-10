from app.tasks.task import Task
import hydra

class TrainTask(Task):

    def __init__(self, data_loader, model, optimizer):
        """
        :type data_loader: app.data.loaders.loader.Loader
        :type model: app.models.model.Model
        :type optimizer: torch.optim.Optimizer
        """
        super().__init__()
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer

    def run(self):
        # TODO
        print('TRAIN')
