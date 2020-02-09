from app.tasks.train import TrainTask
import hydra

@hydra.main(config_path='configs/train.yaml')
def _main(config):
    data_loader = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    optimizer = hydra.utils.instantiate(config.optimizer, model.parameters())
    task = TrainTask(data_loader, model, optimizer)
    task.run()

if __name__ == '__main__':
    _main()
