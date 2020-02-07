def get_task(task_data):
    """
    :type task_data: dict
    :rtype: app.tasks.task.Task
    """
    type = task_data.pop('type')
    if type == 'create_artificial_data':
        from app.tasks.create_artificial_data.task import CreateArtificialDataTask
        return CreateArtificialDataTask(**task_data)
    else:
        raise Execption(f'Unknown task: {type}')
