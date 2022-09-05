import os
import paddle
import paddle.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential

class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.is_train = opt['is_train']        # training or not
        self.scheduler = None                  # scheduler

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        self.scheduler.step(n)

    def current_learning_rate(self):
        return self.scheduler.get_lr()

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.stop_gradient = ~flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.numpy()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pdparams'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        paddle.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        if strict:
            state_dict = paddle.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.set_state_dict(state_dict)
        else:
            state_dict_old = paddle.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.set_state_dict(state_dict)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pdparams'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        paddle.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        # optimizer.set_state_dict(paddle.load(load_path, map_location=lambda storage, loc: storage.cuda(paddle.device.get_device())))
        optimizer.set_state_dict(paddle.load(load_path))

    def update_E(self, decay=0.999):
        netG = self.netG
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k] = netE_params[k]*decay + (netG_params[k])*(1-decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
