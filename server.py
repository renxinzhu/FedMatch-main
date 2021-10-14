from typing import Dict, List, OrderedDict
import gc
import torch
from scipy.spatial import KDTree
from base_model import Backbone, IStateDict
import copy


class Server(Backbone):
    def aggregate(self, state_dict_list: List[Dict[str, OrderedDict[str, torch.Tensor]]]):
        avg_sigma, avg_phi = copy.deepcopy(
            state_dict_list[0]["sigma"]), copy.deepcopy(state_dict_list[0]["phi"])

        for key in avg_sigma:
            sigmas = torch.stack([state_dict["sigma"][key]
                                  for state_dict in state_dict_list])
            avg_sigma[key] = torch.mean(sigmas.type(torch.FloatTensor), dim=0)

        for key in avg_phi:
            phis = torch.stack([state_dict["phi"][key]
                                for state_dict in state_dict_list])
            avg_phi[key] = torch.mean(phis.type(torch.FloatTensor), dim=0)

        # record aggregated params
        self.params["sigma"] = avg_sigma
        self.params["phi"] = avg_phi

    def getHelpers(self, H: int, current_state_dict: OrderedDict[str, torch.Tensor]) -> List[OrderedDict[str, torch.Tensor]]:
        """Get H helpers, returning the clients with the nearest weights

        Args:
            H: numebr of helpers
            current_state_dict: the state_dict of the local client's phi

        Returns:
            List[state_dict]: the state_dict of helpers
        """
        toQuery = flatten_state_dict(current_state_dict).cpu().numpy()
        _, idxes = self.kdtree.query(toQuery, k=H+1)

        return [self.client_phis[i] for i in idxes if i != 0]

    def construct_tree(self, client_phis: List[IStateDict]):
        self.kdtree = None
        gc.collect()

        # load clients parameters
        self.client_phis = client_phis

        # construct kdtree
        flatten_params = [flatten_state_dict(phi).cpu(
        ).numpy() for phi in self.client_phis]
        self.kdtree = KDTree(flatten_params)


def flatten_state_dict(state_dict: IStateDict) -> torch.Tensor:
    phi_flatten = [torch.flatten(tensor.float())
                   for tensor in state_dict.values()]

    return torch.cat(phi_flatten)
