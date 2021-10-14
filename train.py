import random
import gc
import torch
import time
from client import Client
from server import Server
from datasets import generate_random_dataloaders, generate_dataloaders, generate_test_dataloader

batch_size_s = 10
batch_size_u = 100
dataset_name = 'fashionMNIST'

num_clients = 100
local_epochs = 1
rounds = 200
frac_clients = 0.05
num_helpers = 2
h_interval = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hyper_parameters = {
    "lr": 1e-2,
    "wd": 1e-3,  # weight decay
    "lambda_s": 10,  # supervised learning
    "lambda_iccs": 1e-2,
    "lambda_l1": 1e-4,
    "lambda_l2": 10,
}
start_time = time.time()

dataloaders = generate_dataloaders(dataset_name, num_clients, label_ratio=0.05,
                                   iid=True, batch_size_s=batch_size_s, batch_size_u=batch_size_u)
test_dataloader = generate_test_dataloader(dataset_name)
# dataloaders = generate_little_dataloaders(
#     num_clients, label_ratio=0.05, iid=True, size=100)

clients = [Client(dataloader, device=device, client_id=client_id)
           for client_id, dataloader in enumerate(dataloaders)]

# we can load server's params at initialization here
server = Server(dataloader=generate_random_dataloaders(
    dataset_name), device=device)

# load hyper-parameters
for client in clients:
    client.load_hyper_parameters(hyper_parameters)
server.load_hyper_parameters(hyper_parameters)

clients_phi = [client.params['phi'] for client in clients]
server.construct_tree(clients_phi)

for r in range(rounds):
    num_connected = random.sample(
        range(num_clients), int(num_clients * frac_clients))
    server_state_dict = server.params
    server.logger.print(
        f'training clients (round:{r}, connected:{num_connected})')

    # update tree for every 10 rounds
    if r % h_interval == 0:
        clients_phi = [client.params['phi'] for client in clients]
        server.construct_tree(clients_phi)
    state_dict_list = []

    for idx in num_connected:
        gc.collect()
        client = clients[idx]

        helpers_phi = server.getHelpers(
            num_helpers, client.params["phi"])

        loss = client.local_train(state_dict=server_state_dict, helpers_phi=helpers_phi,
                                  epochs=local_epochs, test_dataloader=test_dataloader)

        client_state_dict = client.params
        state_dict_list.append(client_state_dict)

    server.aggregate(state_dict_list)

    test_acc, test_loss = server.evaluate(test_dataloader)

    # save(acc)
    server.logger.print(
        f'aggr_acc:{round(test_acc,4)},aggr_loss:{round(test_loss,4)}')
    server.logger.save_current_state(
        {"round": r, "aggr_acc": test_acc, "aggr_loss": test_loss})

server.logger.print('all clients done')
server.logger.print('server done. ({}s)'.format(time.time()-start_time))
