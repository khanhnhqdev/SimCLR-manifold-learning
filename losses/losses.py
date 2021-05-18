"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()

        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0) # b x n, dim
        anchor = features[:, 0] # b , dim ->  first view

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature # b, b x n
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True) # b, 1
        logits = dot_product - logits_max.detach() # b, b x n

        mask = mask.repeat(1, 2) # b, b x n
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0) # b, b x n

        # print(logits_mask)
        mask = mask * logits_mask # b, b x n
        # print(mask)

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # log(fraction) = log(numerator) - log(denumerator)
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        return loss


class SimCLRManifoldLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, p):
        super(SimCLRManifoldLoss, self).__init__()
        self.p = p
        self.temperature = p['criterion_kwargs']['temperature']
        self.batch_size = p['batch_size']
        self.n_views = p['n_views']

    def forward(self, features, w):
        """
        input:
            - features: hidden feature representation of shape [batch x nviews, dim]

        output:
            - loss: loss computed according to SimCLR
        """

        features = F.normalize(features, dim=1)

        x = []
        for i in range(features.size(0)):
            ids = torch.Tensor([j * self.batch_size + i % self.batch_size for j in range(self.n_views) if j != i // self.batch_size]).long()
            x_i = torch.sum(features[ids] * w[i].view(-1, 1), dim=0)
            x.append(x_i.view(1, -1))
        # x: correspond sum of neighbor of features
        x = torch.cat(x, dim=0)
        x = F.normalize(x, dim=1)

        # label: nviews , batchsize,
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)

        # labels: (nviews x batch) , (nviews x batch),
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(non_blocking=True)

        # cosine similarity matrix: (nviews x batch) , (nviews x batch)
        similarity_matrix = torch.matmul(features, features.T)

        # mask: (nviews x batch) , (nviews x batch)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(non_blocking=True)

        # labels: (nviews x batch) x (nviews x batch - 1)
        labels = labels[~mask].view(labels.shape[0], -1)

        # similarity_matrix: (nviews x batch) x (nviews x batch - 1) cosine similarity matrix
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # positive simCLR + manifold
        positives = (features * x).sum(dim=1).view(-1, 1)
        # negatives: (nviews x batch) x ((nviews x batch) - 2)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits: (nviews x batch) x ((nviews x batch) - 1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(non_blocking=True)

        logits = logits / self.temperature

        CrossEntropyLoss = nn.CrossEntropyLoss()
        loss = CrossEntropyLoss(logits, labels)
        return loss
