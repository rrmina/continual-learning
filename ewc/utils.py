import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Elastic Weight Consolidation")
    parser.add_argument('--num_tasks', type=int, default=1, help='number of tasks')
    parser.add_argument('--num_epochs_per_task', type=int, default=1, help='number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--ewc_train', type=str2bool, default=False, help='ewc train or not')
    parser.add_argument('--ewc_weight', type=float, default=1.0, help='ewc loss hyperparameter')
    args = parser.parse_args()

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