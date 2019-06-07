from torch.nn.init import *


def init(module,types,initializer,init_params=None,category="all"):

    
    def init_func(m):

        if type(m) in types or len(types) == 0:
            if category == "all":
                for param in m.parameters():
                    if init_params == None:

                        initializer(param.data)
                    else:
                        initializer(param.data,**init_params)
            
            elif category == "weight":
                if hasattr(m,"weight"):
                    if init_params == None:

                        initializer(m.weight.data)
                    else:
                        initializer(m.weight.data,**init_params)
            elif category == "bias":
                if hasattr(m,"bias"):
                    if hasattr(m.bias,"data"):
                        if init_params == None:

                            initializer(m.bias.data)
                        else:
                            initializer(m.bias.data,**init_params)

    module.apply(init_func)


def normal_init(module,mean=0,std=1,types=[],category="all"):

    return init(module,types,normal_,{"mean":mean,"std":std},category)

def uniform_init(module,lower=0,upper=1,types=[],category="all"):

    return init(module,types,uniform_,{"lower":lower,"upper":upper},category)

def constant_init(module,value,types=[],category="all"):

    return init(module,types,constant_,{"val":value},category)

def ones_init(module,types=[],category="all"):

    return init(module,types,constant_,{"val":1},category)

def zeros_init(module,types=[],category="all"):

    return init(module,types,constant_,{"val":0},category)

def eye_init(module,types=[],category="all"):

    return init(module,types,eye_,None,category)

def dirac_init(module,types=[],category="all"):

    return init(module,types,dirac_,None,category)


def sparsity_init(module,sparsity_ratio=0.1,std=0.01,types=[],category="all"):

    return init(module,types,sparse_,{"sparsity":sparsity_ratio,"std":std},category)

def kaiming_normal_init(module,neg_slope=0,mode="fan_in",non_linearity="leaky_relu",types=[],category="weight"):

    return init(module,types,kaiming_normal_,{"a":neg_slope,"mode":mode,"nonlinearity":non_linearity},category)

def kaiming_uniform_init(module,neg_slope=0,mode="fan_in",non_linearity="leaky_relu",types=[],category="weight"):

    return init(module,types,kaiming_uniform_,{"a":neg_slope,"mode":mode,"nonlinearity":non_linearity},category)

def xavier_normal_init(module,gain=1,types=[],category="weight"):

    return init(module,types,xavier_normal_,{"gain":gain},category)

def xavier_uniform_init(module,gain=1,types=[],category="weight"):

    return init(module,types,xavier_uniform_,{"gain":gain},category)

def orthorgonal_init(module,gain=1,types=[],category="all"):

    return init(module,types,orthogonal_,{"gain":gain},category)
