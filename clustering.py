from app.tasks.clustering import ClusteringTask
import hydra

@hydra.main(config_path='configs/clustering.yaml')
def _main(config):
    clustering = hydra.utils.instantiate(config.clustering)
    task = ClusteringTask(clustering)
    task.run()

if __name__ == '__main__':
    _main()
