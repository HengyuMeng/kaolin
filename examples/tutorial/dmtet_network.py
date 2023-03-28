import torch
from tqdm import tqdm

# MLP + Positional Encoding
class Decoder(torch.nn.Module):
    """
    __init__方法用于初始化类的属性和网络层，forward方法用于定义网络的前向传播逻辑。
    这个Decoder类的作用是根据输入的三维坐标，输出对应的SDF值和RGB颜色，从而实现可微渲染
    """

    def __init__(self, input_dims = 3, internal_dims = 128, output_dims = 4, hidden = 5, multires = 2):
        """
        input_dims: 输入的维度，默认为3，表示输入是三维坐标。
        internal_dims: 网络中间层的维度，默认为128。
        output_dims: 输出的维度，默认为4，表示输出是一个四元组，包括SDF值和RGB颜色。
        hidden: 网络隐藏层的数量，默认为5。
        multires: 位置编码的频率数量，默认为2。
        __init__方法首先检查是否需要使用位置编码，如果是，则调用get_embedder函数来获取一个嵌入函数和一个输入通道数。
        然后，它使用torch.nn.Linear和torch.nn.ReLU来构建一个多层感知机（MLP），并将其赋值给self.net属性。
        """
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        """
        parameters:
            p: 输入的三维坐标，形状为(batch_size, 3)

        forward方法首先检查是否需要使用位置编码，如果是，则调用self.embed_fn来对输入进行嵌入。
        然后，它使用self.net来对嵌入后的输入进行前向传播，并返回输出。

        return: out是一个torch.Tensor类型的数据，它的形状为(batch_size, output_dims)，
                其中output_dims默认为4，表示输出是一个四元组，包括SDF值和RGB颜色
        """
        if self.embed_fn is not None:
            # 先让顶点的三维坐标通过嵌入函数后，输出一个更高维度的嵌入向量，从而增强网络的表达能力
            p = self.embed_fn(p)

        # 将具有更多信息的嵌入向量作为神经网络的输入，来得到out
        out = self.net(p)
        return out

    def pre_train_sphere(self, iter):
        print ("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            p = torch.rand((1024,3), device='cuda') - 0.5
            ref_value  = torch.sqrt((p**2).sum(-1)) - 0.3
            output = self(p)
            loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())


# Positional Encoding from
# https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    """
    这个类是一个用于实现位置编码的类，它有一个__init__方法和一个embed方法。
    __init__方法用于初始化类的属性和嵌入函数，embed方法用于对输入进行嵌入。
    这个Embedder类的作用是根据输入的三维坐标，输出一个更高维度的嵌入向量，从而增强网络的表达能力
    """
    def __init__(self, **kwargs):
        """
        parameters:
            kwargs: 一个字典，包含以下键值对：
                include_input: 一个布尔值，表示是否在嵌入中包含原始输入，默认为True。
                input_dims: 输入的维度，默认为3，表示输入是三维坐标。
                max_freq_log2: 最大频率的对数，默认为multires-1，表示最大频率是2的multires-1次方。
                num_freqs: 频率的数量，默认为multires，表示使用multires个不同的频率。
                log_sampling: 一个布尔值，表示是否使用对数采样，默认为True。
                periodic_fns: 一个列表，包含周期函数，默认为[torch.sin, torch.cos]。
        """
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        """
        create_embedding_fn方法的作用是根据self.kwargs中的一些参数，来构建一个嵌入函数的列表，
        并将其存储在self.embed_fns属性中，以便在embed方法中使用
        """
        #     create_embedding_fn方法首先创建一个空列表embed_fns，用于存储嵌入函数。
        #     它从self.kwargs中获取’input_dims’，这个键，并将值赋给d变量，并初始化输出维度为0
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # 然后，它检查self.kwargs中是否有’include_input’这个键，如果有，并且其值为True，则表示需要在嵌入中包含原始输入。
        # 如果是这样，则它将一个恒等函数lambda x: x添加到embed_fns中。
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        # 接下来，它从self.kwargs中获取‘max_freq_log2’，‘num_freqs’这些键的值，
        # 并将它们分别赋值给max_freq，N_freqs变量。
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        # 然后，它根据log_sampling的值，来生成一系列的频率带freq_bands。
        # 如果log_sampling为True，则它使用torch.linspace来生成一个从0到max_freq的等差数列，并对其取2的幂；
        # 如果log_sampling为False，则它使用torch.linspace来生成一个从2的0次方到2的max_freq次方的等差数列。
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # 对于freq_bands中的每个频率freq，它遍历periodic_fns列表中的每个周期函数p_fn，
        # 并将一个函数lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)添加到embed_fns中。
        # 最后，它将embed_fns和其长度赋值给self.embed_fns和self.out_dim属性
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """
        inputs: 输入的三维坐标，形状为(batch_size, 3)。
        embed方法使用torch.cat来将self.embed_fns中的每个函数作用于inputs后的结果拼接起来，并返回拼接后的结果。
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
