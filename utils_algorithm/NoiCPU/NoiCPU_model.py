import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_model.NoiCPU import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from .get_partial_label import get_partial_label
from .get_threshold import Threshold

class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x
    
class NoiCPU(nn.Module):
    def __init__(self, args, label_confidence):
        super(NoiCPU, self).__init__()
        self.emam = args.model_m
        self.dataset_name = args.dataset

        self.backbone_online = NoiCPU.get_backbone('resnet18', self.dataset_name)
        self.backbone_target = NoiCPU.get_backbone('resnet18', self.dataset_name)
        
        dim_out, dim_in = self.backbone_online.fc.weight.shape
        dim_mlp = 2048
        self.backbone_online.fc = nn.Identity()
        self.backbone_target.fc = nn.Identity()

        self.projector_online = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        self.projector_target = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))

        self.predictor_online = nn.Sequential(nn.Linear(dim_out, 512), BatchNorm1d(512), nn.ReLU(), nn.Linear(512, dim_out))

        self.encoder_online = nn.Sequential(self.backbone_online, self.projector_online)
        self.encoder_target = nn.Sequential(self.backbone_target, self.projector_target)

        self.classifier = nn.Sequential(self.backbone_online, nn.Linear(dim_in, 2))

        for param_online, param_target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        
        self.register_buffer("prototypes", torch.zeros(2, dim_out))
        self.register_buffer("label_conf", label_confidence)

        self.register_buffer("time_p", torch.tensor(0.5))
        self.register_buffer("p_model", torch.tensor([0.5, 0.5]))
        self.register_buffer("threshold_param", torch.tensor([0.5, 0.5]))
        self.threshold = Threshold(time_p=self.time_p, p_model=self.p_model, class_prior=args.class_prior, momentum=args.threshold_m)

    def forward(self, w_image, s_image_online=None, s_image_target=None, labels=None, args=None, eval_only=False):
        output = self.classifier(w_image)

        if eval_only:
            return output
        
        z1 = self.encoder_online(s_image_online)
        p1 = self.predictor_online(z1)
        p1 = nn.functional.normalize(p1, dim=1)

        # get cls probability
        batchY = get_partial_label(labels).cuda().detach()
        predicted_scores = torch.softmax(output, dim=1) * batchY
        _, pseudo_labels_b = torch.max(predicted_scores, dim=1)

        # compute prototype logits
        prototpyes = self.prototypes.clone().detach()
        prototype_logts = torch.mm(p1, prototpyes.t())
        prototpyes_pro = torch.softmax(prototype_logts, dim=1)

        # update momentum prototypes with pseudo labels
        for feat, label in zip(p1, pseudo_labels_b):
            self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach() # normalize prototypes

        with torch.no_grad():
            for param_online, param_target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
                param_target.data = self.emam * param_target.data + (1.-self.emam) * param_online.data

        z2 = self.encoder_target(s_image_target)
        z2 = nn.functional.normalize(z2, dim=1)

        return p1, z2, output, prototpyes_pro
    
    def confidence_update(self, temp_un_conf, batch_index, labels, conf_ema_m):
        with torch.no_grad():
            batchY = get_partial_label(labels).cuda().detach()
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.label_conf[batch_index, :] = conf_ema_m * self.label_conf[batch_index, :] + (1 - conf_ema_m) * pseudo_label

    def threshold_cls_update(self, cls_logits, labels):
        self.threshold_param = self.threshold.get_threshold(cls_logits, labels, softmax=True)
        self.time_p = self.threshold.time_p
        self.p_model = self.threshold.p_model

    # def threshold_proto_update(self, proto_pro, labels):
    #     self.threshold_param = self.threshold.get_threshold(proto_pro, labels, softmax=False)
    #     self.time_p = self.threshold.time_p
    #     self.p_model = self.threshold.p_model

    @staticmethod
    def get_backbone(backbone_name, dataset_name):
        return {'resnet18': ResNet18(dataset_name=dataset_name),
                'resnet34': ResNet34(dataset_name=dataset_name),
                'resnet50': ResNet50(dataset_name=dataset_name),
                'resnet101': ResNet101(dataset_name=dataset_name),
                'resnet152': ResNet152(dataset_name=dataset_name)}[backbone_name]