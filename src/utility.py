import torch, os, dropbox  #pip install dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
import pandas as pd
# import torch.optim as optim
import argparse
from io import StringIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import inspect, re
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import subprocess
import torch.optim as optim

# from optimizers.TR import TR
# from optimizers.APTS_D import APTS_D
# from optimizers.APTS import APTS_W
from models.neural_networks import *



def flatten_tensor(tensor):
    """
    Flattens a PyTorch tensor into a 2D tensor with the first dimension as the number of samples
    and the second dimension as the product of the remaining dimensions.
    
    Args:
    - tensor: a PyTorch tensor with shape (n_sample, a, b, c, ...)
    
    Returns:
    - flattened_tensor: a PyTorch tensor with shape (n_sample, a*b*c*...)
    """
    n_sample = tensor.shape[0]
    flattened_dim = torch.tensor(tensor.shape[1:]).prod().item()
    flattened_tensor = tensor.view(n_sample, flattened_dim)
    return flattened_tensor



def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size



def get_optimizer_fun(optimizer_name, op=None):
    if optimizer_name == "SGD":
        return optim.SGD
    elif optimizer_name == "Adam":
        return optim.Adam
    elif optimizer_name == "LBFGS":
        return optim.LBFGS
    elif "TR" in optimizer_name:
        return TR
    elif "APTS" in optimizer_name and 'D' in optimizer_name and not '2' in optimizer_name:
        return APTS_D
    elif "APTS" in optimizer_name and 'D' in optimizer_name and '2' in optimizer_name:
        return APTS_D_2
    elif "APTS" in optimizer_name and 'W' in optimizer_name:
        return APTS_W
    else:
        print("Optimizer not recognized.")
        exit(1)



def get_net_fun_and_params(dataset, net_nr):
    if dataset == "MNIST":
        if net_nr == 0:
            net = MNIST_FCNN
            net_params = {"hidden_sizes": [32, 32]}
        elif net_nr == 1:
            net = MNIST_FCNN
            net_params = {"hidden_sizes": [512, 256]}
        elif net_nr == 2:
            net = MNIST_CNN
            net_params = {}
    elif dataset == "CIFAR10":
        if net_nr == 0:
            net = CIFAR_FCNN
            net_params = {"hidden_sizes": [128, 64]}
        elif net_nr == 1:
            net = CIFAR_FCNN
            net_params = {"hidden_sizes": [512, 512]}
        elif net_nr == 2:
            net = CIFAR_CNN
            net_params = {}
        elif net_nr == 3:
            net = CifarNet
            net_params = {}
        elif net_nr == 4:
            net = SmallResNet
            net_params = {}
    # else:
    #     net_params = {'hidden_sizes':[32, 32], 'input_size': 50, 'output_size': 50}
    #     net = NetSine

    return net, net_params



def prepare_distributed_environment():
    cuda = torch.cuda.is_available()
    if cuda:
        device_count = torch.cuda.device_count()
        print(f'Setting cuda "device_count" to {device_count}')
    else:
        device_count = 1
        print(f"Using CPU only.")

    device_id = 0
    if cuda:
        try:
            # Cluster
            os.environ["MASTER_PORT"] = "29501"
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = os.environ["SLURM_NODEID"]
            node_list = os.environ["SLURM_NODELIST"]
            master_node = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1"
            )
            os.environ["MASTER_ADDR"] = master_node

            dist.init_process_group(backend="nccl")
            device_id = dist.get_rank()
            print(f"Device id: {device_id}")
        except:
            # Or the IP address/hostname of the master node
            os.environ['MASTER_ADDR'] = 'localhost'
            # A free port on the master node
            os.environ['MASTER_PORT'] = '12355'
            # The total number of GPUs in the distributed job
            os.environ['WORLD_SIZE'] = str(device_count)
            # The unique identifier for this process (0-indexed)
            os.environ['RANK'] = '0'
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            # dist.init_process_group(backend='gloo') #nccl not worning on home pc



# Collect command-line arguments
def parse_args():
    # Default value is selected when no command line argument of the same name is provided
    parser = argparse.ArgumentParser(description='PyTorch multi-gpu training.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--trials', type=int, default=2, help='Number of experiment trials.')
    parser.add_argument('--net_nr', type=int, default=3, help="Model number (NOT number of models).")
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Dataset name. Currently, MNIST and CIFAR10 are supported.')
    parser.add_argument('--minibatch_size', type=int, default=50000, help='Batch size for training (default 100).')
    parser.add_argument('--overlap_ratio', type=float, default=0, help='Overlap ratio for minibatches.')
    parser.add_argument('--optimizer_name', type=str, default="Adam", help="Main optimizer to be used.")
    # Generic optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0, help="Momentum value for SGD optimizer/momentum (True, False) for TR/APTS.")
    # For Adam/TR/APTS
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam optimizer.")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam optimizer.")
    # TR/APTS specific parameter settings
    parser.add_argument('--radius', type=float, default=0.01, help='TR radius.')
    parser.add_argument('--max_radius', type=float, default=4.0, help='Maximum learning rate.')
    parser.add_argument('--min_radius', type=float, default=0.001, help='Minimum learning rate.')
    parser.add_argument('--decrease_factor', type=float, default=0.5, help='Learning rate decrease factor.')
    parser.add_argument('--increase_factor', type=float, default=2.0, help='Learning rate increase factor.')
    parser.add_argument('--is_adaptive', type=int, default=0, help='Choice of whether optimizer is adaptive or not.')
    parser.add_argument('--second_order', type=int, default=0, help='Choice of whether to use second order information or not.')
    parser.add_argument('--second_order_method', type=str, default="SR1", help="TR second-order strategy.")
    parser.add_argument('--delayed_second_order', type=int, default=0, help="How many epochs to wait before using second-order information.")
    parser.add_argument('--accept_all', type=int, default=0, help='Choice of whether to accept all steps produced by local optimizer or not.')
    parser.add_argument('--acceptance_ratio', type=float, default=0.75, help='Trust-region model acceptance ratio.')
    parser.add_argument('--reduction_ratio', type=float, default=0.25, help='Trust-region model reduction ratio.')
    parser.add_argument('--history_size', type=int, default=5, help="How many vectors to keep track of for second-order information.")
    parser.add_argument('--norm_type', type=int, default=2, help="Norm type for TR radius computation.")
    # APTS specific parameter settings
    parser.add_argument('--local_optimizer', type=str, default="TR", help="Local optimizer to be used.")
    parser.add_argument('--max_iter', type=int, default=5, help='Numer of optimizer iterations per micro-batch.')
    parser.add_argument('--global_pass', type=int, default=1, help="Choice of whether to do global pass or not.")
    parser.add_argument('--foc', type=int, default=1, help="Choice of whether to have first-order consistency or not.")
    parser.add_argument('--nr_models', type=int, default=1, help="Number of models to be used in APTS.")
    parser.add_argument('--counter', type=int, default=0, help="Counter for filename in case of parallel execution.")
    parser.add_argument('--op', type=str, default="sum", help="APTS reduction on local steps to be performed.")
    default_db_path = "./results/blah.db" 
    parser.add_argument('--db_path', type=str, default=default_db_path, help="Local optimizer to be used.")

    args = parser.parse_args()
    if "SGD" not in args.optimizer_name:
        args.momentum = True if (args.momentum != 0) else False

    # Convert relevant arguments to boolean type after parsing
    args.is_adaptive = bool(args.is_adaptive)
    args.second_order = bool(args.second_order)
    args.accept_all = bool(args.accept_all)
    args.global_pass = bool(args.global_pass)
    args.foc = bool(args.foc)
    return args

        
### YOU SHOULD ADD ALL OPTIMIZERS YOU WISH TO SUPPORT IN THE FOLLOWING FUNCTIONS ###
def check_args_are_valid(args):
    # Check that common arguments are properly set
    # lower case args.dataset
    if args.dataset.lower() not in ['mnist', 'cifar10', 'sine']:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    if args.optimizer_name.lower() not in ['sgd', 'adam', 'lbfgs', 'tr'] and 'apts' not in args.optimizer_name.lower():
        raise ValueError(f"Optimizer {args.optimizer_name} not recognized.")
    if args.epochs < 0 or not isinstance(args.epochs, int):
        raise ValueError('Number of epochs must be a positive integer.')
    if args.trials < 0 or not isinstance(args.trials, int):
        raise ValueError('Number of trials must be a positive integer.')
    if args.net_nr < 0 or not isinstance(args.net_nr, int):
        raise ValueError('Model number must be a positive integer.') 
    if args.minibatch_size < 0 or not isinstance(args.minibatch_size, int):
        raise ValueError('Minibatch size must be a positive integer.')
    if args.overlap_ratio < 0 or args.overlap_ratio > 1:
        raise ValueError('Overlap ratio must be between 0 and 1.')

    if 'sgd' in args.optimizer_name.lower():
        # Check that SGD arguments are properly set
        if args.lr < 0:
            raise ValueError('Learning rate must be positive.')
        if args.momentum < 0 or args.momentum > 1:
            raise ValueError('Momentum must be between 0 and 1.')
        
    elif 'adam' in args.optimizer_name.lower():
        # Check that Adam arguments are properly set
        if args.lr < 0:
            raise ValueError('Learning rate must be positive.')
        if args.beta1 < 0 or args.beta1 > 1:
            raise ValueError('Beta1 must be between 0 and 1.')
        if args.beta2 < 0 or args.beta2 > 1:
            raise ValueError('Beta2 must be between 0 and 1.')

    elif 'lbfgs' in args.optimizer_name.lower():
        # Check that LBFGS arguments are properly set
        if args.lr < 0:
            raise ValueError('Learning rate must be positive.')
        if args.max_iter < 0 or not isinstance(args.max_iter, int):
            raise ValueError('Number of iterations must be a positive integer.')
        if args.history_size < 0 or not isinstance(args.history_size, int):
            raise ValueError('History size must be a positive integer.')
        
    elif 'tr' in args.optimizer_name.lower() or 'apts' in args.optimizer_name.lower():
        # Check that TR arguments are properly set
        if not isinstance(args.momentum, bool):
            raise ValueError('Momentum must be either True or False.')
        if not isinstance(args.is_adaptive, bool):
            raise ValueError('Adaptive must be either True or False.')
        if not isinstance(args.second_order, bool):
            raise ValueError('Second order must be either True or False.')
        if not isinstance(args.accept_all, bool):
            raise ValueError('Accept all must be either True or False.')
        if not isinstance(args.global_pass, bool):
            raise ValueError('Global pass must be either True or False.')
        if not isinstance(args.foc, bool):
            raise ValueError('FOC must be either True or False.')

        if args.radius < 0 or args.radius > args.max_radius or args.radius < args.min_radius:
            raise ValueError('Radius must be positive, \leq than max radius, and \geq than min radius.')
        if args.min_radius < 0 or args.min_radius > args.radius:
            raise ValueError('Minimum radius must be positive.')
        if args.max_radius < 0 or args.max_radius < args.radius:
            raise ValueError('Maximum radius must be positive.') 
        if args.decrease_factor < 0 or args.decrease_factor > 1:
            raise ValueError('Decrease factor must be between 0 and 1.')
        if args.increase_factor <= 1:
            raise ValueError('Increase factor must be greater than 1.')
        if args.second_order_method.lower() not in ['sr1', 'bfgs']:
            raise ValueError('Second-order method not recognized.')
        if args.delayed_second_order < 0 or not isinstance(args.delayed_second_order, int):
            raise ValueError('Delayed second-order must be positive.')
        if args.acceptance_ratio < 0 or args.acceptance_ratio > 1:
            raise ValueError('Acceptance ratio must be between 0 and 1.')
        if args.reduction_ratio < 0 or args.reduction_ratio > 1:
            raise ValueError('Reduction ratio must be between 0 and 1.')
        if args.history_size < 0 or not isinstance(args.history_size, int):
            raise ValueError('History size must be positive.')
        if args.beta1 < 0 or args.beta1 > 1:
            raise ValueError('Beta1 must be between 0 and 1.')
        if args.beta2 < 0 or args.beta2 > 1:
            raise ValueError('Beta2 must be between 0 and 1.')
        if args.norm_type not in [1, 2, 2e308]: # 2e308 will be our way of saying inf
            raise ValueError('Norm type not recognized.')

        # Check that APTS_D/APTS_W arguments are properly set
        if args.optimizer_name in ['APTS_D', 'APTS_W']:
            if args.max_iter < 0 or not isinstance(args.max_iter, int):
                raise ValueError('Number of iterations must be positive.')
            if args.nr_models < 0 or not isinstance(args.nr_models, int): # args.nr_models should be set outside by the user 
                raise ValueError('World size must be positive.')
            
            if args.op.lower() not in ["avg", "sum"]:
                raise ValueError('Reduction operation not recognized.')
            else:
                if args.op.lower() == "avg" and args.norm_type != 2e308:
                    raise ValueError('Norm type must be inf when using average reduction.')
                    
def get_optimizer_params(args):
    optimizer_params = {}
    check_args_are_valid(args) # this will check depending on the optimizer name
    if 'SGD' in args.optimizer_name:
        optimizer_params['lr'] = args.lr
        optimizer_params['momentum'] = args.momentum

    elif 'Adam' in args.optimizer_name:
        optimizer_params['lr'] = args.lr
        optimizer_params['betas'] = (args.beta1, args.beta2)

    elif 'LBFGS' in args.optimizer_name:
        optimizer_params['lr'] = args.lr
        optimizer_params['max_iter'] = args.max_iter
        optimizer_params['history_size'] = args.history_size
        optimizer_params['line_search_fn'] = 'strong_wolfe'

    elif 'TR' in args.optimizer_name:
        optimizer_params['radius'] = args.radius
        optimizer_params['max_radius'] = args.max_radius
        optimizer_params['min_radius'] = args.min_radius
        optimizer_params['decrease_factor'] = args.decrease_factor
        optimizer_params['increase_factor'] = args.increase_factor
        optimizer_params['is_adaptive'] = args.is_adaptive
        optimizer_params['second_order'] = args.second_order
        optimizer_params['second_order_method'] = args.second_order_method
        optimizer_params['delayed_second_order'] = args.delayed_second_order # epoch in which second-order information should start being considered
        optimizer_params['device'] = args.device
        # Accept all updates. If False it is the standard TR which recomputes the loss untill it decreases
        optimizer_params['accept_all'] = args.accept_all
        optimizer_params['acceptance_ratio'] = args.acceptance_ratio
        optimizer_params['reduction_ratio'] = args.reduction_ratio
        optimizer_params['history_size'] = args.history_size
        optimizer_params['momentum'] = args.momentum # Adam Momentum
        optimizer_params['beta1'] = args.beta1 # Parameter for Adam momentum
        optimizer_params['beta2'] = args.beta2 # Parameter for Adam momentum
        optimizer_params['norm_type'] = torch.inf if args.norm_type == 2e308 else args.norm_type

    elif 'apts' in args.optimizer_name.lower() and 'd' in args.optimizer_name.lower():
        optimizer_params['device'] = args.device
        optimizer_params['global_opt'] = TR
        optimizer_params['local_opt'] = TR
        optimizer_params['max_iter'] = args.max_iter
        optimizer_params['global_pass'] = args.global_pass
        optimizer_params['foc'] = args.foc
        if '2' in args.optimizer_name.lower():
            optimizer_params['op'] = args.op
        optimizer_params['nr_models'] = args.nr_models

        optimizer_params['global_opt_params'] = {
            'radius': args.radius,
            'max_radius': args.max_radius,
            'min_radius': args.min_radius,
            'decrease_factor': args.decrease_factor,
            'increase_factor': args.increase_factor,
            'is_adaptive': args.is_adaptive, 
            'second_order': args.second_order,
            'second_order_method': args.second_order_method,
            'delayed_second_order': args.delayed_second_order,
            'accept_all': args.accept_all,
            'acceptance_ratio': args.acceptance_ratio,
            'reduction_ratio': args.reduction_ratio,
            'history_size': args.history_size,
            'momentum': args.momentum, # Adam Momentum
            'beta1': args.beta1, # Parameter for Adam momentum
            'beta2': args.beta2, # Parameter for Adam momentum
            'norm_type': torch.inf if args.norm_type == 2e308 else args.norm_type,            
        }
        
        optimizer_params['local_opt_params'] = optimizer_params['global_opt_params'].copy()
        if 'avg' in args.op.lower():
            optimizer_params['local_opt_params']['radius'] = optimizer_params['global_opt_params']['radius']
        else:
            optimizer_params['local_opt_params']['radius'] = optimizer_params['global_opt_params']['radius'] / args.nr_models
    
    elif 'apts' in args.optimizer_name.lower() and 'w' in args.optimizer_name.lower():
        optimizer_params['device'] = args.device
        optimizer_params['global_opt'] = TR
        optimizer_params['local_opt'] = TR
        optimizer_params['max_iter'] = args.max_iter
        optimizer_params['global_pass'] = args.global_pass
        optimizer_params['nr_models'] = args.nr_models

        optimizer_params['global_opt_params'] = {
            'radius': args.radius,
            'max_radius': args.max_radius,
            'min_radius': args.min_radius,
            'decrease_factor': args.decrease_factor,
            'increase_factor': args.increase_factor,
            'is_adaptive': args.is_adaptive, 
            'second_order': args.second_order,
            'second_order_method': args.second_order_method,
            'delayed_second_order': args.delayed_second_order,
            'accept_all': args.accept_all,
            'acceptance_ratio': args.acceptance_ratio,
            'reduction_ratio': args.reduction_ratio,
            'history_size': args.history_size,
            'momentum': args.momentum, # Adam Momentum
            'beta1': args.beta1, # Parameter for Adam momentum
            'beta2': args.beta2, # Parameter for Adam momentum
            'norm_type': torch.inf, # TODO: change in case different norm desired            
        }

        optimizer_params['local_opt_params'] = optimizer_params['global_opt_params'].copy()
        optimizer_params['local_opt_params']['radius'] = optimizer_params['global_opt_params']['radius'] / 10 # TODO: change this if something else desired
        optimizer_params['local_opt_params']['min_radius'] = 0.0

    return dict(sorted(optimizer_params.items()))




def get_ignore_params(optimizer_name):
    ignore_params = ["params"] # this is the case for all optimizers
    if "APTS" in optimizer_name or "TR" in optimizer_name:
        ignore_params.append("device") 
    if "APTS" in optimizer_name:
        ignore_params.append("model")
        ignore_params.append("loss_fn")
        
    return ignore_params





def Email_sender(OBJ):
    Gmail_Account={'Username': 'grodt.prj@gmail.com', 'Password': 'dusskqzgvvrnsong'}
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(Gmail_Account['Username'], Gmail_Account['Password']) 
    for Er in ['samuel.adolfo.cruz.alegria@usi.ch', 'ken13889@msn.com']:
        message = f"Subject: {OBJ}\nFrom: {Gmail_Account['Username']}\nTo: {Er}\n\n"
        server.sendmail(Gmail_Account['Username'], Er, message)
    server.quit()



def get_string_from_class(obj):
    class_obj = obj.__class__
    
    # Get the source code
    source_code = inspect.getsource(class_obj)
    
    # Remove single line comments
    source_code = re.sub(r"#.*$", "", source_code, flags=re.MULTILINE)
    
    # Remove multiline comments
    source_code = re.sub(r'""".*?"""', '', source_code, flags=re.DOTALL)
    source_code = re.sub(r"'''.*?'''", '', source_code, flags=re.DOTALL)
    source_code = '\n'.join([line for line in source_code.split('\n') if line.strip() != ''])
    return source_code
    
