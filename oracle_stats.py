from app.data.loaders import get_loader
from app.tasks.oracle_stats import OracleStatsTask
import hydra

@hydra.main(config_path='configs/oracle_stats.yaml')
def _main(config):
    loader = get_loader(config.loader)
    task = OracleStatsTask(loader)
    task.run()

if __name__ == '__main__':
    _main()
