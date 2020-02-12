from app.tasks.create_artificial_data import CreateArtificialDataTask
import hydra

@hydra.main(config_path='configs/create_artificial_data.yaml')
def _main(config):
    task = CreateArtificialDataTask(
        config.train_size,
        config.val_size,
        config.test_size,
        config.max_depth,
        config.max_determiners,
        config.max_nouns,
        config.max_particles,
        config.max_verbs,
        config.save_dir
    )
    task.run()

if __name__ == '__main__':
    _main()
