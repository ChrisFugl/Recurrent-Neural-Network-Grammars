from app.data.loaders import get_loader
from app.tasks.create_oracle import CreateOracleTask
import hydra

@hydra.main(config_path='configs/create_oracle.yaml')
def _main(config):
    loader = get_loader(config.loader)
    task = CreateOracleTask(loader, config.loader.data_dir, config.type, config.fine_grained_unknowns, config.unknown_non_terminals)
    task.run()

if __name__ == '__main__':
    _main()
