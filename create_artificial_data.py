import hydra

@hydra.main(config_path='configs/create_artificial_data.yaml')
def _main(config):
    task = hydra.utils.instantiate(config)
    task.run()

if __name__ == '__main__':
    _main()
