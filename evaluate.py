from app.models.load import load_saved_model
from app.tasks.evaluate.evaluate import EvaluateTask
from app.utils import get_device, set_seed
import hydra
import json

@hydra.main(config_path='configs/evaluate.yaml')
def _main(config):
    assert config.model_dir is not None, 'Please specify model_dir.'
    assert config.samples is not None, 'Please specify samples.'
    set_seed(config.seed)
    device = get_device(config.gpu)
    model, generative, action_converter, token_converter, tag_converter, non_terminal_converter = load_saved_model(device, config.model_dir)
    samples_path = hydra.utils.to_absolute_path(config.samples)
    with open(samples_path, 'r') as file:
        samples = json.load(file)
    task = EvaluateTask(device, model, generative, action_converter, token_converter, tag_converter, non_terminal_converter, samples)
    task.run()

if __name__ == '__main__':
    _main()
