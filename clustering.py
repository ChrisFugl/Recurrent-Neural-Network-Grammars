from app.clustering import get_clustering
from app.data.loaders import get_loader
from app.tasks.clustering import ClusteringTask
import hydra

@hydra.main(config_path='configs/clustering.yaml')
def _main(config):
    output_dir = config.loader.data_dir
    loader = get_loader(config.loader)
    clustering = get_clustering(config.clustering, loader, output_dir)
    task = ClusteringTask(clustering)
    task.run()

if __name__ == '__main__':
    _main()
