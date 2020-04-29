from app.samplers import get_sampler
from app.tasks.sample import SampleTask
from app.utils import get_device, set_seed
import hydra

@hydra.main(config_path='configs/sample.yaml')
def _main(config):
    assert not config.iterator.shuffle, 'iterator is not allowed to shuffle data'
    set_seed(config.seed)
    device = get_device(config.gpu)
    sampler = get_sampler(device, config.data, config.iterator, config.sampler)
    task = SampleTask(device, config.data, sampler)
    task.run()

if __name__ == '__main__':
    _main()
