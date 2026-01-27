import torch

class Threshold:
    def __init__(self, time_p, p_model, class_prior, momentum, clip_thresh=True):
        self.m = momentum
        self.class_prior = class_prior
        self.clip_thresh = clip_thresh

        self.time_p = time_p.cuda()
        self.p_model = p_model.cuda()

    @torch.no_grad()
    def get_threshold(self, batch_input, labels, softmax):

        if softmax:
            input = torch.softmax(batch_input, dim=1)
        else:
            input = batch_input

        u_idx = torch.where(labels == 1)[0]
        input = input[u_idx].clone().detach()

        if self.class_prior is not None:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(input[:, 1], self.class_prior, interpolation="nearest")
            
            return torch.tensor([1 - self.time_p, self.time_p])
        
        else:
            max_probs, _ = torch.max(input, dim=-1, keepdim=True)

            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
            self.time_p = torch.clip(self.time_p, 0.0, 0.8)

            self.p_model = self.p_model * self.m + (1 - self.m) * input.mean(dim=0)
            threshold = self.p_model / torch.max(self.p_model, dim=-1)[0] * self.time_p

            return threshold