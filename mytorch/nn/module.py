class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

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
        from mytorch.tensor import Tensor
        if isinstance(value, Tensor):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()

    def children(self):
        return self._modules.values()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        # 返回模型所有参数的 dict：{ "layer.weight": Tensor }
        state = {}

        def _get_state(module, prefix=''):
            for name, param in module._parameters.items():
                state[prefix + name] = param
            for name, child in module._modules.items():
                _get_state(child, prefix + name + '.')

        _get_state(self)
        return state

    def load_state_dict(self, state_dict):
        def _load_state(module, prefix=''):
            for name, param in module._parameters.items():
                full_name = prefix + name
                if full_name in state_dict:
                    param.data[...] = state_dict[full_name].data
                else:
                    raise KeyError(
                        f"Parameter '{full_name}' not found in state_dict")
            for name, child in module._modules.items():
                _load_state(child, prefix + name + '.')

        _load_state(self)

    def save(self, file_path):
        import pickle
        state = self.state_dict()
        # 只保存 param.data，避免保存 Tensor 对象本身
        to_save = {k: v.data for k, v in state.items()}
        # 确保文件路径的目录存在
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 使用 pickle 保存数据
        with open(file_path, 'wb') as f:
            pickle.dump(to_save, f)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        import pickle
        from mytorch.tensor import Tensor
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        # 将 numpy 数据转换为 Tensor
        state_dict = {k: Tensor(v) for k, v in loaded.items()}
        self.load_state_dict(state_dict)
        print(f"Model loaded from {file_path}")
