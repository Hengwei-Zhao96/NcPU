import time

import faiss
import numpy as np
import torch
import torch.nn as nn


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    start_time = time.time()
    print("performing kmeans clustering")
    results = {"im2cluster": [], "centroids": [], "density": []}

    num_cluster = args.num_cluster
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = 20
    clus.nredo = 5
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = args.gpu
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
    density = args.temperature * density / density.mean()

    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results["centroids"] = centroids
    results["density"] = density
    results["im2cluster"] = im2cluster

    print("Kmeans end. Eplapsed {} s".format(time.time() - start_time))

    return results
