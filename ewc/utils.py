import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Elastic Weight Consolidation")
    parser.add_argument('--num_tasks', type=int, default=10, help='number of tasks')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=300, help='batch size')
    parser.add_argument('--ewc_train', type=str2bool, default=False, help='ewc train or not')
    parser.add_argument('--ewc_weight', type=float, default=1e4, help='ewc loss hyperparameter')
    parser.add_argument('--custom_suffix', type=str, default="", help='custom name suffix')
    parser.add_argument('--permute_indices', default=None, type=str, help="permutation file")
    parser.add_argument('--hidden_size', default=256, type=int, help="hidden size")
    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')