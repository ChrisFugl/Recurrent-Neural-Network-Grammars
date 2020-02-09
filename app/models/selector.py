def get_model(model_data):
    """
    :type model_data: dict
    :rtype: app.models.model.Model
    """
    type = model_data.pop('type')
    raise Execption(f'Unknown model: {type}')
