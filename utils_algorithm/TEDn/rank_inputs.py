import numpy as np
import torch


def rank_inputs(net, u_trainloader, alpha, u_size):

    net.eval()
    output_probs = np.zeros(u_size)
    keep_samples = np.ones_like(output_probs)
    true_targets_all = np.zeros(u_size)

    with torch.no_grad():
        for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
            idx = idx.numpy()

            inputs = inputs.cuda()
            outputs = net(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 0]
            output_probs[idx] = probs.detach().cpu().numpy().squeeze()
            true_targets_all[idx] = true_targets.numpy().squeeze()

    sorted_idx = np.argsort(output_probs)

    keep_samples[sorted_idx[u_size - int(alpha * u_size) :]] = 0

    neg_reject = np.sum(true_targets_all[sorted_idx[u_size - int(alpha * u_size) :]] == 1.0)

    neg_reject = neg_reject / int(alpha * u_size)
    return keep_samples
