def metric_prin(metric):
    prin = []
    for key, value in metric.items():
        prin.append(key + ":{:.2f}".format(value))
    prin = "\t".join(prin) + "\n"
    return prin
