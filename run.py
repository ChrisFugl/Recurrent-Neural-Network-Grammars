from app.tasks.selector import get_task
import os
import sys
import yaml
import yamale
from yamlinclude import YamlIncludeConstructor

SCHEMA_PATH = 'app/schema.yaml'

def _run():
    arguments = sys.argv
    if len(arguments) != 2:
        print('Usage: python run.py [path to task file]', file=sys.stderr)
        return
    task_path = arguments[1]
    if not os.path.exists(task_path):
        print(f'"{task_path}" does not exist', file=sys.stderr)
        return
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.SafeLoader, base_dir='.')
    schema = yamale.make_schema(SCHEMA_PATH)
    task_data = yamale.make_data(task_path)
    yamale.validate(schema, task_data, strict=True)
    task = get_task(task_data[0][0]['task'])
    task.run()

if __name__ == '__main__':
    _run()
