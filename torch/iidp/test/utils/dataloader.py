from torch.iidp.config.examples.config_utils import NUM_DATASET, REGISTERED_MODELS_FOR_DATASET


class MockDataLoader(object):
    def __init__(self, model_name):
        if model_name not in REGISTERED_MODELS_FOR_DATASET:
            raise ValueError(
                f'argument ```model_name``` must be chosen among {REGISTERED_MODELS_FOR_DATASET}')
        self.dataset = MockDataset(model_name)

class MockDataset(object):
    def __init__(self, model_name):
        if model_name not in REGISTERED_MODELS_FOR_DATASET:
            raise ValueError(
                f'argument ```model_name``` must be chosen among {REGISTERED_MODELS_FOR_DATASET}')
        self.model_name = model_name

    def __len__(self):
        return NUM_DATASET(self.model_name)