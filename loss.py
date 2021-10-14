from typing import Dict, List, cast, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ITorchReturnTypeMax = NamedTuple('torch_return_type_max', [(
    'indices', torch.Tensor), ('values', torch.Tensor)])


def _icc_loss(pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    kl_loss_helper = nn.KLDivLoss(reduction="batchmean")
    _sum = 0.0

    for helper_pred in helper_preds:
        _sum += kl_loss_helper(pred, helper_pred).float()

    return _sum / len(helper_preds)


def _transform_onehot(tensor: torch.Tensor) -> torch.Tensor:
    max_values = cast(torch.Tensor, torch.max(
        tensor, dim=1, keepdim=True).values)
    return (tensor >= max_values).float() - \
        torch.sigmoid(tensor - max_values).detach() + \
        torch.sigmoid(tensor - max_values)


def _calculate_pseudo_label(local_pred: torch.Tensor, helper_preds: List[torch.Tensor]):
    _sum = torch.zeros_like(local_pred)
    for pred in [local_pred, *helper_preds]:
        one_hot = _transform_onehot(pred)
        _sum += one_hot

    return torch.argmax(_sum, dim=1)


def _consistency_regularization(pred: torch.Tensor, pred_noised: torch.Tensor, helper_preds: List[torch.Tensor]):
    pseudo_label = _calculate_pseudo_label(
        pred_noised, helper_preds).type(torch.LongTensor).to(device)

    pseudo_label_CE_loss = F.cross_entropy(
        pred_noised, pseudo_label)
    kl_loss = _icc_loss(pred, helper_preds)

    return pseudo_label_CE_loss + kl_loss


def unsupervised_loss(sigma: Dict[str, torch.Tensor], phi: Dict[str, torch.Tensor], pred: torch.Tensor,
                      pred_noised: torch.Tensor, helper_preds: List[torch.Tensor], lambda_l1: int, lambda_l2: int, lambda_iccs: int):
    # flatten params
    sigma_cat = torch.cat([torch.flatten(tensor.float())
                           for tensor in sigma.values()])
    phi_cat = torch.cat([torch.flatten(tensor.float())
                         for tensor in phi.values()])

    iccs_loss = _consistency_regularization(pred, pred_noised, helper_preds)
    l1_target = torch.zeros_like(phi_cat)
    return iccs_loss * lambda_iccs + F.mse_loss(sigma_cat, phi_cat) * lambda_l2 - F.l1_loss(phi_cat, l1_target) * lambda_l1
