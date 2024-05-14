MODEL_TITLE_MAP = {
    'resnet18_cifar10': 'ResNet-18 on CIFAR10',
    'resnet50': 'ResNet-50 on ImageNet',
    'rcnn': 'Faster-R-CNN on MS-COCO',
    'gnmt': 'GNMT on WMT16',
    'bert': 'BERT-base on SQuAD v1.1',
    'gpt2': 'GPT-2 on WikiText-2',
    'vit': 'ViT on ImageNet'
}

REGISTERED_MODELS_FOR_DATASET = ['resnet50', 'vit', 'rcnn', 'gnmt', 'resnet18_cifar10']

DATASET_NAME_MAP = {
    'resnet50': 'imagenet',
    'vit': 'imagenet',
    'rcnn': 'coco',
    'gnmt': 'wmt16',
    'resnet18_cifar10': 'cifar10'
}


def NUM_DATASET(model_name):
    dataset_num = {
        'imagenet': 1281167,
        'coco': 117266,
        'wmt16': 3498161,
        'cifar10': 50000
    }
    return dataset_num[DATASET_NAME_MAP[model_name]]
