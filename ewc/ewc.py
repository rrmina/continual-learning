import torch
import torch.nn as nn
from copy import deepcopy

class EWC(object):
    def __init__(self, model, loaders, curr_task, device, empirical_fisher=True):
        self.model = deepcopy(model)
        self.loaders = loaders
        self.curr_task = curr_task
        self.device = device
        self.empirical_fisher = empirical_fisher
        self.criterion = nn.CrossEntropyLoss()
        self.theta_prev = {}
        self.diag_fisher = {}

        # Consolidate
        self._consolidate()
        self.loaders = None
        del self.model

    def _consolidate(self):
        # Store Previous theta values
        # and initialize diag fishers to 0
        for n, p in self.model.named_parameters():
            if (p.requires_grad == True):
                self.theta_prev[n] = p.data.clone()
                self.diag_fisher[n] = p.data.clone().zero_()

        # Compute Diagonal fishers
        for i in range(self.curr_task + 1):
            # Zero out model gradients
            for p in self.model.parameters():
                p.grad = None

            # Compute new gradients
            # Square of sums instead of sums of squares
            for x, y in self.loaders[i]:
                batch_size = x.shape[0]
                x, y = x.view(batch_size, -1).to(self.device), y.to(self.device)
                out = self.model(x)
                if (self.empirical_fisher):  # Empirical Fisher
                    preds = out.max(1)[1]
                    loss = self.criterion(out, preds)
                else:                   # Exact Fisher
                    loss = self.criterion(out, y)
                
                loss.backward()

                # Record the diagonal fishers
                for n, p in self.model.named_parameters():
                    if (p.requires_grad == True):
                        self.diag_fisher[n] += p.grad.detach() ** 2
                break

    def ewc_loss(self, model):
        losses = []
        for n, p in model.named_parameters():
            if (p.requires_grad == True):
                losses.append((self.diag_fisher[n] * (self.theta_prev[n] - p) ** 2).sum())

        return sum(losses)