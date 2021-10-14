from typing import Any, Dict, OrderedDict
from logger import Logger
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import copy

IStateDict = OrderedDict[str, torch.Tensor]

LR_DECAY_PATIENCE = 5


class Backbone:
    def __init__(self, dataloader: Dict[str, DataLoader], device: torch.device,
                 sigma: IStateDict = None, phi: IStateDict = None, client_id: int = None, psi_factor=0.2):
        super().__init__()
        self.client_id = client_id
        self.logger = Logger(client_id=client_id)
        self.device = device
        self.dataloader = dataloader
        self.params = {
            "sigma": sigma if sigma else create_model().state_dict(),
            "phi": phi if phi else create_model().state_dict()
        }
        if phi is None:
            for key in self.params["phi"]:
                self.params["phi"][key] = self.params["sigma"][key] * psi_factor

        self.lr_decay_patience = LR_DECAY_PATIENCE
        self.lowest_vloss = float('inf')

    def adjust_lr(self, vloss: float):
        if vloss < self.lowest_vloss:
            self.lr_decay_patience = LR_DECAY_PATIENCE
            self.lowest_vloss = vloss
            return

        self.lr_decay_patience -= 1
        if self.lr_decay_patience == 0:
            self.hyper_parameters['lr'] *= self.hyper_parameters['wd']
            self.lr_decay_patience = LR_DECAY_PATIENCE

    def load_hyper_parameters(self, hyper_parameters: Dict[str, Any]):
        self.hyper_parameters = copy.deepcopy(hyper_parameters)

    def evaluate(self, dataloader: DataLoader):
        correct = 0
        running_loss = 0
        model = NetDecomposed(
            freezed=self.params['sigma'], unfreezed=self.params['phi'])
        loss_fn = torch.nn.CrossEntropyLoss()
        model.eval()

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                running_loss += loss_fn(pred, y).item() * X.size(0)

        test_acc = correct / len(dataloader.dataset)
        test_loss = running_loss / len(dataloader.dataset)

        return test_acc, test_loss


# def create_model():
#     # return nn.ModuleList([nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
#     #                       # return nn.ModuleList([nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#     #                       nn.MaxPool2d(kernel_size=2, stride=2),
#     #                       nn.ReLU(),

#     #                       nn.Conv2d(16, 32, kernel_size=5,
#     #                                 stride=1, padding=2),
#     #                       nn.Conv2d(32, 32, kernel_size=5,
#     #                                 stride=1, padding=2),
#     #                       nn.MaxPool2d(kernel_size=2, stride=2),
#     #                       nn.ReLU(),

#     #                       nn.Conv2d(32, 32, kernel_size=5,
#     #                                 stride=1, padding=2),
#     #                       nn.Conv2d(32, 32, kernel_size=5,
#     #                                 stride=1, padding=2),
#     #                       nn.MaxPool2d(kernel_size=2, stride=2),
#     #                       nn.ReLU(),

#     #                       nn.Flatten(),
#     #                       nn.Linear(4*4*32, 10),
#     #                       # nn.Linear(32*7*7, 10),
#     #                       ])
#     return nn.ModuleList([nn.Conv2d(3, 32, 3, 1, padding=1),
#                           nn.Conv2d(32, 64, 3, 1, padding=1),
#                           nn.ReLU(),
#                           nn.MaxPool2d(2, 2),

#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.ReLU(),
#                           nn.MaxPool2d(2, 2),

#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.ReLU(),
#                           nn.MaxPool2d(2, 2),

#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.Conv2d(64, 64, 3, 1, padding=1),
#                           nn.ReLU(),
#                           nn.MaxPool2d(4, 1),

#                           nn.Flatten(),
#                           nn.Linear(64, 10),
#                           ])


class NetDecomposed(nn.Module):
    def __init__(self, freezed: IStateDict, unfreezed: IStateDict):
        super().__init__()
        self.freezed = create_model()
        self.unfreezed = create_model()

        self.freezed.load_state_dict(freezed)
        self.unfreezed.load_state_dict(unfreezed)

    def forward(self, X):
        out = X
        for i in range(len(self.freezed)):
            if isinstance(self.freezed[i], (nn.Conv2d, nn.Linear)):
                out = self.freezed[i](out) + self.unfreezed[i](out)
            else:
                out = self.unfreezed[i](out)
        return out

    def get_freezed(self):
        return copy.deepcopy(self.freezed.state_dict())

    def get_unfreezed(self):
        return copy.deepcopy(self.unfreezed.state_dict())

######################################Resnet-9#######################################


def create_model():
    return nn.Sequential(nn.Conv2d(1, 32, 3, 1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(32, 64, 3, 1, padding=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),

                         ###
                         nn.Sequential(
        nn.Conv2d(64, 64, 3, 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, padding=1),
        nn.ReLU()),

        nn.Conv2d(64, 128, 3, 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        ###
        nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU()),

        nn.MaxPool2d(3, 1),

        nn.Flatten(),
        nn.Linear(256, 10),
    )


class NetDecomposed(nn.Module):
    def __init__(self, freezed: IStateDict, unfreezed: IStateDict):
        super().__init__()
        self.freezed = create_model()
        self.unfreezed = create_model()

        self.freezed.load_state_dict(freezed)
        self.unfreezed.load_state_dict(unfreezed)

    def forward(self, X):
        out = X
        for i in range(len(self.freezed)):
            if isinstance(self.freezed[i], nn.Sequential):
                out2 = out
                for j in range(len(self.freezed[i])):
                    if self._contain_params(self.freezed[i][j]):
                        out2 = self.freezed[i][j](
                            out2) + self.unfreezed[i][j](out2)
                    else:
                        out2 = self.unfreezed[i][j](out2)
                out = out + out2
            elif self._contain_params(self.freezed[i]):
                out = self.freezed[i](out) + self.unfreezed[i](out)
            else:
                out = self.unfreezed[i](out)
        return out

    def get_freezed(self):
        return copy.deepcopy(self.freezed.state_dict())

    def get_unfreezed(self):
        return copy.deepcopy(self.unfreezed.state_dict())

    def _contain_params(self, layer):
        param_list = list(layer.parameters())
        if len(param_list) != 0:
            return True
        else:
            return False
