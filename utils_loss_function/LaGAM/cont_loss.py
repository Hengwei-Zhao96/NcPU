import torch
import torch.nn as nn


class ContLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        cont_cutoff=False,
        knn_aug=False,
        num_neighbors=0,
        contrastive_clustering=1,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_clustering = contrastive_clustering
        self.cont_cutoff = cont_cutoff
        self.knn_aug = knn_aug
        self.num_neighbors = num_neighbors

    def forward(self, q, k, cluster_idxes=None, preds=None, start_knn_aug=False):
        batch_size = q.shape[0]

        q_and_k = torch.cat([q, k], dim=0)
        l_i = torch.einsum("nc,kc->nk", [q, q_and_k]) / self.temperature

        self_mask = torch.ones_like(l_i, dtype=torch.float)
        self_mask = torch.scatter(self_mask, 1, torch.arange(batch_size).view(-1, 1).cuda(), 0).detach().cuda()

        positive_mask_i = torch.zeros_like(l_i, dtype=torch.float)
        positive_mask_i = (
            torch.scatter(
                positive_mask_i,
                1,
                batch_size + torch.arange(batch_size).view(-1, 1).cuda(),
                1,
            )
            .detach()
            .cuda()
        )

        l_i_exp = torch.exp(l_i)
        l_i_exp_sum = torch.sum((l_i_exp * self_mask), dim=1, keepdim=True)

        loss = -torch.sum(torch.log(l_i_exp / l_i_exp_sum) * positive_mask_i, dim=1).mean()

        if cluster_idxes is not None and self.contrastive_clustering:
            cluster_idxes = cluster_idxes.view(-1, 1)
            cluster_idxes_kq = torch.cat([cluster_idxes, cluster_idxes], dim=0)
            mask = torch.eq(cluster_idxes, cluster_idxes_kq.T).float().cuda()

            if self.cont_cutoff:
                preds = preds.detach()
                pred_labels = (preds > 0.5) * 1
                pred_labels = pred_labels.view(-1, 1)
                pred_labels_kq = torch.cat([pred_labels, pred_labels], dim=0)
                label_mask = torch.eq(pred_labels, pred_labels_kq.T).float().cuda()

                mask = mask * label_mask

            if self.knn_aug and start_knn_aug:
                cosine_corr = q @ q_and_k.T
                _, kNN_index = torch.topk(cosine_corr, k=self.num_neighbors, dim=-1, largest=True)
                mask_kNN = torch.scatter(torch.zeros(mask.shape).cuda(), 1, kNN_index, 1)
                mask = ((mask + mask_kNN) > 0.5) * 1

            mask = mask.float().detach().cuda()
            batch_size = q.shape[0]
            anchor_dot_contrast = torch.div(torch.matmul(q, q_and_k.T), self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).cuda(), 0)
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            loss_prot = -mean_log_prob_pos.mean()
            loss += loss_prot

        return loss
