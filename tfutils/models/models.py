#TO DO : SUMMARY, ONNX EXPORT, LIBTORCH EXPORT, LOADING, SAVING

from torch.nn.parallel.data_parallel import DataParallel
import torch
import copy
from ..fp16.fp16 import MultiSequential,Convert


def clip_grads(model,lower,upper):
    """

    :param model:
    :param lower:
    :param upper:
    :return:
    """
    for params in model.parameters():
        params.data.clamp_(lower,upper)

def save_model(model,path,save_architecture=False):

    if type(model) == DataParallel:
        model = model.module

    if isinstance(model,MultiSequential):
        
        for child in model.children():

            if not isinstance(child,Convert):
                model = child 
                break

    model = copy.deepcopy(model).float().cpu()

    if save_architecture:
        torch.save(model, path)
    else:
        state = model.state_dict()
        torch.save(state, path)

def load_model(model,path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(checkpoint)
    except:
        copy = dict()
        for x, y in zip(model.state_dict(), checkpoint):
            new_name = y[y.index(x):]
            copy[new_name] = checkpoint[y]
        model.load_state_dict(copy)


def model_summary(model,*input_tensors,item_length=26):
    """

    :param model:
    :param input_tensors:
    :param item_length:
    :param tensorboard_log:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple("Layer", ["name", "input_size", "output_size","num_parameters","multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):

            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0
            if hasattr(module,"weight"):
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size()))) * torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()

            elif isinstance(module, nn.Linear):

                flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)).item()

            summary.append(
                ModuleDetails(name=layer_name, input_size=list(input[0].size()), output_size=list(          output.size()), num_parameters=params, multiply_adds=flops))

        if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and not isinstance(module,tofp16)  and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = "Model Summary" + os.linesep + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
        ' ' * (space_len - len("Name")), ' ' * (space_len - len("Input Size")),
        ' ' * (space_len - len("Output Size")), ' ' * (space_len - len("Parameters")),
        ' ' * (space_len - len("Multiply Adds (Flops)"))) + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        details += "{}{}{}{}{}{}{}{}{}{}".format(layer.name, ' ' * (space_len - len(layer.name)), layer.input_size,
                                                 ' ' * (space_len - len(str(layer.input_size))), layer.output_size,
                                                 ' ' * (space_len - len(str(layer.output_size))),
                                                 layer.num_parameters,
                                                 ' ' * (space_len - len(str(layer.num_parameters))),
                                                 layer.multiply_adds,
                                                 ' ' * (space_len - len(str(layer.multiply_adds)))) + os.linesep + '-' * space_len * 5 + os.linesep


    details += os.linesep + "Total Parameters: {}".format(params_sum) + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {}".format(flops_sum) + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])


    return details