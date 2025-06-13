class Module:
    from mytorch.tensor import Tensor

    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params += module.parameters()
        return params

    def add_parameter(self, name, tensor):
        self._parameters[name] = tensor

    def add_module(self, name, module):
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
