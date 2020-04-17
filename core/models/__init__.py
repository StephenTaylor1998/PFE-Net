from core.models.icnet import get_icnet


def get_segmentation_model(model, **kwargs):
    models = {

        'icnet': get_icnet,

    }
    return models[model](**kwargs)