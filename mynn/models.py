from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    CNN 模型，用于 MNIST 分类。
    结构: conv2D(1,8,3,p=1) -> ReLU -> MaxPool2D -> conv2D(8,16,3,p=1) -> ReLU -> MaxPool2D
          -> Flatten -> Linear(784,128) -> ReLU -> Linear(128,10)
    """
    def __init__(self):
        self.layers = [
            conv2D(1, 8, 3, padding=1), ReLU(), MaxPool2D(),           # (batch,1,28,28) -> (batch,8,14,14)
            conv2D(8, 16, 3, padding=1), ReLU(), MaxPool2D(),          # (batch,8,14,14) -> (batch,16,7,7)
            Flatten(),                                                  # (batch,16,7,7) -> (batch,784)
            Linear(784, 128), ReLU(),
            Linear(128, 10),
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            saved = pickle.load(f)
        structure = saved[0]  # 第一项是层类型列表，描述网络结构
        # 按结构描述重建每一层并加载权重
        self.layers = []
        param_idx = 1
        for layer_type in structure:
            if layer_type == 'conv2D':
                layer = conv2D(**saved[param_idx]['init_args'])
                layer.W = saved[param_idx]['W']
                layer.b = saved[param_idx]['b']
                layer.params = {'W': layer.W, 'b': layer.b}
                layer.weight_decay = saved[param_idx].get('weight_decay', False)
                layer.weight_decay_lambda = saved[param_idx].get('lambda', 0)
                self.layers.append(layer)
                param_idx += 1
            elif layer_type == 'ReLU':
                self.layers.append(ReLU())
            elif layer_type == 'MaxPool2D':
                self.layers.append(MaxPool2D())
            elif layer_type == 'Flatten':
                self.layers.append(Flatten())
            elif layer_type == 'Linear':
                layer = Linear(1, 1)  # 占位维度，权重稍后覆盖
                layer.W = saved[param_idx]['W']
                layer.b = saved[param_idx]['b']
                layer.params = {'W': layer.W, 'b': layer.b}
                layer.weight_decay = saved[param_idx].get('weight_decay', False)
                layer.weight_decay_lambda = saved[param_idx].get('lambda', 0)
                self.layers.append(layer)
                param_idx += 1

    def save_model(self, save_path):
        # 保存结构描述 + 每层参数到 pickle 文件
        structure = []
        params = [structure]
        for layer in self.layers:
            if isinstance(layer, conv2D):
                structure.append('conv2D')
                params.append({
                    'W': layer.params['W'], 'b': layer.params['b'],
                    'init_args': {
                        'in_channels': layer.in_channels,
                        'out_channels': layer.out_channels,
                        'kernel_size': layer.kernel_size,
                        'stride': layer.stride,
                        'padding': layer.padding,
                    },
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
            elif isinstance(layer, Linear):
                structure.append('Linear')
                params.append({
                    'W': layer.params['W'], 'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
            elif isinstance(layer, ReLU):
                structure.append('ReLU')
            elif isinstance(layer, MaxPool2D):
                structure.append('MaxPool2D')
            elif isinstance(layer, Flatten):
                structure.append('Flatten')
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)