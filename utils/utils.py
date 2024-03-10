import argparse
import logging
import random
import torch
import os
import numpy
import json
import logging
import pickle
from scipy import stats

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist

from p2p_server.rpc.rpc_client import readJsonc, loadConfig, to_namespace

SIMULATION_MODE = "simulation"
STANDALONE_MODE = "standalone"

BTPS_TRANSFER_MODE = "BTPS"
PS_TRANSFER_MODE = "PS"

FEDAVG_STRATEGY = "fedavg" # random selection
OORT_STRATEGY = "oort"
FedP2P_STRATEGY = "fedp2p"

def str2bool(v):
    if isinstance(v, bool):
        return v
    while True:
        if v.startswith('='):
            v = v[1:]
        else:
            break
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError(None, f'Value {v} should be true/false, True/False or yes/no.')


def check_args(args):
    if args.client_selection:
        if args.selected_clients_number == -1 and args.selected_clients_proportion == 0:
            raise argparse.ArgumentError(None, "When using client selection, the number of selected clients in each iteration must be set manually.")
        # This if-else means selected_clients_number has higher priority than selected_clients_proportion.
        if args.selected_clients_number >= 1:
            pass
        else:
            args.selected_clients_number = int(args.selected_clients_proportion * (args.world_size - 1))
    else:
        args.selected_clients_number = args.world_size - 1

    if args.selected_clients_number < 1 or args.selected_clients_number > args.world_size - 1:
        raise argparse.ArgumentError(None, "Check selected_clients_number and selected_clients_proportion again.")


def get_args():
    parser = argparse.ArgumentParser(description="Train models on Imagenette under ASGD")
    parser.add_argument("--model", type=str, default="resnet18", help="The job's name.")
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--world_size", type=int, default=3,
                        help="Total number of workers including the parameter server and clients.")
    default_data_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.dirname(default_data_dir)
    default_data_dir = os.path.dirname(default_data_dir)
    default_data_dir = os.path.join(default_data_dir, "data")
    parser.add_argument("--data_dir", type=str,
                        default=default_data_dir, help="The location of dataset.")
    parser.add_argument("--dataset", type=str,
                        default="cifar10", help="Dataset.")
    parser.add_argument("--num_classes", type=int,
                        default=10, help="number of classes in your dataset.")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="29600", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of each worker during training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--local_epoch", type=int, default=1, help="Number of local epochs.")
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--num_evaluate_threads", type=int, default=1,
                        help="parameter server uses certain number of threads to evaluate the model.")
    parser.add_argument("--log_level", type=str, default="DEBUG",
                        help="debug level: NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL, CRITICAL")
    parser.add_argument("--aggregate_frequency", type=int, default=5,
                        help="aggregate the gradients every certain number of iterations in one epoch")
    parser.add_argument("--running_mode", type=str, default=SIMULATION_MODE, 
                        choices=[SIMULATION_MODE, STANDALONE_MODE],
                        help="set the running mode. simulation and standalone are available.")
    parser.add_argument("--transfer_mode", type=str, default=BTPS_TRANSFER_MODE, 
                        choices=[BTPS_TRANSFER_MODE, PS_TRANSFER_MODE],
                        help="set the transfer mode. 1) PS using torch.distributed 2) BTPS using torch.distributed to transfer control message and bit-torrent to transfer data.")
    parser.add_argument("--client_selection", type=str2bool, default=False,
                        help="use client selection or not. Default is False.")
    parser.add_argument("--client_selection_strategy", type=str, default="fedavg",
                        choices=[FEDAVG_STRATEGY, OORT_STRATEGY, FedP2P_STRATEGY],
                        help="client selection strategy. Default is fedavg.")
    parser.add_argument("--selected_clients_number", type=int, default=-1,
                        help="number of selected clients in each iteration.")
    parser.add_argument("--selected_clients_proportion", type=float, default=0,
                        help="proportion of selected clients in each iteration.")
    parser.add_argument(
        "--exploration_proportion", type=float, default=0.5,
        help="Select # clients to explore their utilities, and exploit the remaining clients' utilities. Only valid when using oort as the client selection strategy."
    )
    parser.add_argument(
        "--exploration_proportion_decay", type=float, default=0.98,
        help="After each round, exploration_proportion = exploration_proportion * exploration_proportion_decay. It's only valid in oort."
    )
    parser.add_argument(
        "--exploration_proportion_min", type=float, default=0.2,
        help="After each round, exploration_proportion decays. But it will not be smaller than exploration_proportion_min. It's only valid in oort."
    )
    parser.add_argument(
        "--cut_off_util", type=float, default=0.05,
        help="Make the selection boundary wider. Oort doesn't simply select top K and it uses probability to select from a range. It's only valid in oort."
    )
    parser.add_argument("--use_gpu", type=str2bool, default=False,
                        help="use gpu or not. Default is False. In simulation mode, it's better to use cpu to simulate more clients as gpu memory is smaller than host memory.")
    parser.add_argument(
        "--shard_size", type=int, default=300, 
        help="Only valid when using MNIST dataset."
    )
    parser.add_argument(
        "--mnist_iid", type=str2bool, default=False, 
        help="Use iid or non-iid MNIST dataset. Only valid when using MNIST dataset."
    )
    parser.add_argument(
        "--rate_limit", type=str2bool, default=False, 
        help="In simulation mode, whether to limit the upload and download rate. If it's True, the rate will be determined by the client's profile. It's only valid in simulation mode."
    )
    parser.add_argument(
        "--round_penalty", type=float, default=2.0, 
        help="In oort, round_penalty is used to penalize the clients' system utility if the clients' round completion time is longer than the round_prefer_duration. It's only valid in oort."
    )
    parser.add_argument(
        "--test_interval", type=int, default=5,
        help="test interval"
    )
    args = parser.parse_args()
    check_args(args)

    return args


def get_logger(args, name):
    # 2023-08-02 19:23:44 [0] "/home/whr/fs_gnn/dist_gcn_train.py", line 844, DEBUG: torch.Size([512, 500])

    # note: 如果是多进程，对logging模块的初始化需要在每个进程中都运行一次
    # 并且，logging模块是线程安全的，但并不是进程安全的
    # 如果要保证进程安全，需要将其它进程的消息汇总到一个进程，然后由同一进程中的某些logger（标识进程）来完成
    log_format = f'%(asctime)s %(name)s "%(pathname)s", line %(lineno)d, %(levelname)s: %(message)s\n'

    assert args.log_level in logging._nameToLevel.keys(
    ), "log_level should be one of [NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL, CRITICAL]"

    handler = logging.StreamHandler()
    handler.setLevel(logging._nameToLevel[args.log_level])
    formatter = logging.Formatter(fmt=log_format)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[args.log_level])
    logger.addHandler(handler)

    return logger


def get_updated_config_file(RANK:int, master_addr: str, master_port: int, model: str, dataset: str):
    """
        load dict object from jsonc
        
        update the dict object by params and datetime
        
        return the updated dict object and the path of the updated json file
    """
    current_path = os.path.abspath(__file__)
    current_path = os.path.dirname(current_path)
    current_path = os.path.dirname(current_path)

    jsonc_config_path = os.path.join(current_path, "rpc", "rpc_server", "config.jsonc")
    json_config_path = os.path.join(current_path, "rpc", "rpc_server", "config.json")

    # update config.json according to the datetime
    config = json.loads(readJsonc(jsonc_config_path))
    config['server']['ServerIP'] = master_addr
    config['server']['ServerPort'] = int(master_port)
    config['model']['ModelPath'] = os.path.join(
        config['model']['ModelPath'], model, dataset, get_datetime_str())
    if os.environ.get('StackName'):
        config['tracker']['URLList'] = [
            [
                f"udp://{os.environ['StackName']}_tracker:6969/announce",
            ],
        ]

    return config, json_config_path


def update_config_file(RANK: int, master_addr: str, master_port: int, model: str, dataset: str, logger: logging.Logger):
    if RANK == 0:
        config, json_config_path = get_updated_config_file(RANK, master_addr, master_port, model, dataset)
        with open(json_config_path, 'w') as f:
            f.write(json.dumps(config))

        # DONE: transfer the updated config.json instead of relying on NFS
        config_json = json.dumps(config)
        config_bytes = config_json.encode()
        config_tensor = torch.from_numpy(numpy.frombuffer(config_bytes, numpy.uint8))
        # size
        dist.broadcast(torch.tensor([config_tensor.shape[0]], dtype=torch.int64), 0)
        # data
        dist.broadcast(config_tensor, 0)
        config = to_namespace(config)
    else:
        # size
        config_size = torch.empty(1, dtype=torch.int64)
        dist.broadcast(config_size, 0)
        # data
        config_tensor = torch.empty(config_size[0], dtype=torch.uint8)
        dist.broadcast(config_tensor, 0)
        config_bytes = config_tensor.numpy().tobytes()
        config_json = config_bytes.decode()
        config = json.loads(config_json)
        config = to_namespace(config)

    logger.info(f"config {config}")
    logger.info(
        f"master_addr {config.server.ServerIP}:{config.server.ServerPort}, model_root_dir {config.model.ModelPath}")
    return config


def get_datetime_str():
    return datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")


def split_dataset(dataset: Dataset, N: int, seed: int):
    dataset_length = len(dataset)
    index = list(range(dataset_length))
    index_list = [[]]*N
    random.seed(seed)
    random.shuffle(index)

    piece_length = int(dataset_length/N)
    start = 0
    datasets = [[]]*N

    for i in range(N):
        index_list[i] = index[start:start+piece_length]
        start += piece_length
        feat = []
        label = []
        for j in index_list[i]:
            tmp = dataset[j]
            if tmp[0].shape[0] == 1:
                feat.append(tmp[0].squeeze())
            else:
                feat.append(tmp[0])
            label.append(tmp[1])
        feat = torch.stack(feat)
        label = torch.tensor(label)
        datasets[i] = [feat, label]

    return datasets


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def GetPearsonCorrelationCoefficient():
    data = load_client_profile()
    computation = []
    communication = []
    for value in data.values():
        computation.append(value['computation'])
        communication.append(value['communication'])

    # Pearson correlation coefficient: 0.0024169218848259294
    # P-value: 0.08744723051476527
    correlation, p_value = stats.pearsonr(computation, communication)
    print("Pearson correlation coefficient:", correlation)
    print("P-value:", p_value)


def GetSpearmanCorrelationCoefficient():
    data = load_client_profile()
    computation = []
    communication = []
    for value in data.values():
        computation.append(value['computation'])
        communication.append(value['communication'])

    # Spearman correlation coefficient: 0.003023359711877614
    # P-value: 0.03252991845126464
    correlation, p_value = stats.spearmanr(computation, communication)
    print("Spearman correlation coefficient:", correlation)
    print("P-value:", p_value)


def broadcast_unfixed_length_tensor(tensor: torch.Tensor, src: int, group: dist.group):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    if RANK == src:
        if tensor is None:
            raise ValueError
        # size
        size = torch.tensor([tensor.shape[0]], dtype=torch.int64)
        dist.broadcast(size, src, group=group)
        # data
        dist.broadcast(tensor, src, group=group)
    else:
        # size
        size = torch.empty(1, dtype=torch.int64)
        dist.broadcast(size, src, group=group)
        # data
        tensor = torch.empty(size[0], dtype=torch.uint8)
        dist.broadcast(tensor, src, group=group)

    return tensor

def load_client_profile():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "client_device_capacity")
    with open(path, "rb") as f:
        res = pickle.load(f)

    # # 500000
    # print(len(res))

    # keys = [1, 4,5,7,8,3463,34635]
    # for key in keys:
    #     print(key,res[key])
    # # 1 {'computation': 153.0, 'communication': 2209.615982329485}
    # # 4 {'computation': 149.0, 'communication': 13507.696000153657}
    # # 5 {'computation': 29.0, 'communication': 6924.407283130328}
    # # 7 {'computation': 176.0, 'communication': 32545.573620752573}
    # # 8 {'computation': 44.0, 'communication': 42360.068898122656}
    # # 3463 {'computation': 21.0, 'communication': 11154.383933690891}
    # # 34635 {'computation': 82.0, 'communication': 82504.44631466508}

    # GetPearsonCorrelationCoefficient()
    # # Pearson correlation coefficient: 0.0024169218848259294
    # # P-value: 0.08744723051476527
    # GetSpearmanCorrelationCoefficient()
    # # Pearson correlation coefficient: 0.003023359711877614
    # # P-value: 0.03252991845126464

    return res


def generate_client_profiles(world_size:int):
    client_profile = load_client_profile()
    res = []
    # server's profile is included.
    for index in random.sample(range(1, len(client_profile)), world_size):
        res.append(client_profile[index])

    return res


def python_object_to_tensor(data, reverse=False):
    """
        python object and torch.Tensor are transformed into each other.
    """
    if not reverse:
        # python object to pytorch tensor
        data_bytes = pickle.dumps(data)
        data_tensor = torch.from_numpy(numpy.frombuffer(data_bytes, numpy.uint8))
        return data_tensor
    else:
        # python object to pytorch tensor
        assert type(data) is torch.Tensor
        data_bytes = data.numpy().tobytes()
        data_object = pickle.loads(data_bytes)
        return data_object


def calculate_statistical_utility(losses: torch.Tensor, sample_number) -> float:
    """
        $|B_i| \sqrt {\\frac {1}{|B_i|} \sum _{k\in{B_i}} {LOSS(k)^2}}$
    """
    return sample_number * torch.sqrt(torch.sum(torch.pow(losses, 2)) / losses.shape[0]).item()


if __name__ == "__main__":
    args = get_args()
    print(args)
    logger = get_logger(args, "test")
    logger.info("test")

    train_dataset, train_dataloader, test_dataset, test_dataloader = load_dataset(
        args.dataset, args.data_dir, args.batch_size)

    # print(train_dataset)
    # print(test_dataset)
    # split_datasets = split_dataset(train_dataset, 7, 0)
    # for feature, label in split_datasets:
    #     print(feature.shape, label.shape)

    print(args.batch_size, len(train_dataset))
    i = 0
    for x, y in train_dataloader:
        print(x.shape)
        i += 1
    print(i, int(len(train_dataset)/args.batch_size))
