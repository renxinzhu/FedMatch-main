import copy
from typing import Dict, List, cast
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from base_model import IStateDict, NetDecomposed, Backbone
from loss import unsupervised_loss

CONFIDENCE_THRESHOLD = 0.75


class Client(Backbone):
    def local_train(self, helpers_phi: List[IStateDict],
                    state_dict: Dict[str, IStateDict], epochs: int, test_dataloader: DataLoader):
        self.params["sigma"] = copy.deepcopy(state_dict["sigma"])
        self.params["phi"] = copy.deepcopy(state_dict["phi"])
        self.helpers_phi = helpers_phi

        trained_sigma = self.__train_supervised(epochs=epochs)
        trained_phi = self.__train_unsupervised(epochs=epochs)

        self.params["sigma"] = trained_sigma
        self.params["phi"] = trained_phi

        test_acc, test_loss = self.evaluate(test_dataloader)
        self.logger.print(
            f"test_acc:{round(test_acc,4)},test_loss:{round(test_loss,4)},lr:{self.hyper_parameters['lr']}")

        self.adjust_lr(test_acc)

        return test_loss

    def __train_supervised(self, epochs=100):
        model = NetDecomposed(
            freezed=self.params["phi"], unfreezed=self.params["sigma"]).to(self.device)
        opt = torch.optim.SGD(model.unfreezed.parameters(),
                              lr=self.hyper_parameters['lr'], momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for __, X, y in self.dataloader['labeled']:
                model.train()
                X = X.to(self.device)
                y = y.to(self.device)

                pred = model(X)
                loss = loss_fn(pred, y) * self.hyper_parameters["lambda_s"]
                opt.zero_grad()
                loss.backward()
                opt.step()

        return model.get_unfreezed()

    def __train_unsupervised(self, epochs: int):
        model = NetDecomposed(
            freezed=self.params["sigma"], unfreezed=self.params["phi"]).to(self.device)
        opt = torch.optim.SGD(model.unfreezed.parameters(),
                              lr=self.hyper_parameters['lr'], momentum=0.9)
        loss_fn = unsupervised_loss

        for _ in range(epochs):
            for noised_X, X, __ in self.dataloader['unlabeled']:
                model.train()

                X = X.to(self.device)
                noised_X = noised_X.to(self.device)

                output = model(X)
                pred = F.softmax(output, dim=-1)

                max_values = cast(torch.Tensor, torch.max(pred, dim=1).values)
                confident_idxes = [
                    idx for idx, value in enumerate(max_values.tolist()) if value > CONFIDENCE_THRESHOLD]

                if not confident_idxes:
                    # skip this minibatch of unsupervised learning
                    # if all our prediction confidence < CONFIDENCE_THRESHOLD
                    continue

                confident_pred = pred[confident_idxes]
                confident_X = X[confident_idxes]

                print('confident preds', confident_pred)

                confident_noised_X = noised_X[confident_idxes]

                noised_pred = model(confident_noised_X)
                helper_preds = self.__helper_predictions(confident_X)

                loss = loss_fn(self.params["sigma"], self.params["phi"], confident_pred,
                               noised_pred, helper_preds, self.hyper_parameters["lambda_l1"],
                               self.hyper_parameters["lambda_l2"], self.hyper_parameters["lambda_iccs"])

                opt.zero_grad()
                loss.backward()
                opt.step()

        return model.get_unfreezed()

    def __helper_predictions(self, X: torch.Tensor):
        '''make prediction one by one instead of creating all in the RAM to reduce RAM usage'''
        with torch.no_grad():
            helpers_pred = []
            for helper_phi in self.helpers_phi:
                model = NetDecomposed(
                    freezed=self.params["sigma"], unfreezed=helper_phi).to(self.device)
                pred = model(X)
                helpers_pred.append(pred)
            return helpers_pred
