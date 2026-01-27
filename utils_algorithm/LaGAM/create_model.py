from utils_model.LaGAM.meta_resnet import ResNetMeta


def create_model(num_class, dataset_name):
    model = ResNetMeta(num_class=num_class, dataset_name=dataset_name)
    return model
