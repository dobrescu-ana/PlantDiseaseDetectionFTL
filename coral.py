import torch

def coral_loss(source, target):
    d = source.size(1)
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    loss = torch.mean(torch.mul((source_cov - target_cov), (source_cov - target_cov)))
    return loss / (4 * d * d)

def compute_covariance(feature):
    n = feature.size(0)
    feature_mean = torch.mean(feature, dim=0, keepdim=True)
    feature_centered = feature - feature_mean
    cov = torch.mm(feature_centered.t(), feature_centered) / (n - 1)
    return cov