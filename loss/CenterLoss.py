import torch
from torch import nn

class CenterLoss(nn.Module):
    def __init__(self,nCenters,featureSize, alpha) -> None:
        super(CenterLoss,self).__init__()
        self.alpha = alpha
        self.register_buffer('centers', (
                torch.rand(nCenters, featureSize) - 0.5) * 2)


    def forward(self,features,targets,mode='train'):
        features = features.view(features.size(0), -1)
        target_centers = self.centers[targets]
        criterion = torch.nn.MSELoss()
        center_loss = criterion(features, target_centers)
        if mode == 'train':
            centerDelta = self.getCenterDelta(features,targets)
            self.centers -= centerDelta
        return center_loss


    def getCenterDelta(self,features, targets):
        # implementation equation (4) in the center-loss paper
        features = features.view(features.size(0), -1)
        targets, indices = torch.sort(targets)
        target_centers = self.centers[targets]
        features = features[indices]

        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(
                targets.cpu(), sorted=True, return_inverse=True)

        uni_targets = uni_targets
        indices = indices

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).index_add_(0, indices, delta_centers)

        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(
                targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
                1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(
                targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

        delta_centers = delta_centers / (same_class_feature_count + 1.0) * self.alpha
        result = torch.zeros_like(self.centers)
        result[uni_targets, :] = delta_centers
        return result
