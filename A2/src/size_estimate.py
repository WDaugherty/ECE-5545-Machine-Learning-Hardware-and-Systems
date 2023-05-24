import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            flops['bias'] = 0
            if isinstance(module, nn.Linear):
                # Source: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                # TODO: fill-in (start)
                if module.bias!= None:
                    flops['bias'] = module.bias.size()[0] * batch_size
                flops['weight'] = 2 * np.prod(module.weight.size()) * batch_size
                # TODO: fill-in (end)

            if isinstance(module, nn.Conv2d):
                 # Source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                # TODO: fill-in (start) 
                if module.bias != None:
                    flops['bias'] = np.prod(output[0].shape) * batch_size
                layer_shape = list(module.parameters())[0].size()
                flops['weight'] = np.prod(output[0].shape) * 4 * layer_shape[-1] * layer_shape[-2] * batch_size
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                 # Source: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TODO: fill-in (end)
                flops['bias'] = 0
                flops['weight'] = 0
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # Source: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
                # TODO: fill-in (end)
                flops['bias'] = 0
                flops['weight'] = 0
                # TODO: fill-in (end)
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
     
    # TODO: fill-in (start)
    return sum([np.prod(layer.size()) for layer in list(model.parameters())])
    # TODO: fill-in (end)


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
   
    # TODO: fill-in (start)
    model = model.to(device)
    input_vec = np.prod(input_shape)
    var = torch.ones(input_shape)
    output_vec = np.prod(model(var).size())
    return (input_vec + output_vec) * 4
    # TODO: fill-in (end)

