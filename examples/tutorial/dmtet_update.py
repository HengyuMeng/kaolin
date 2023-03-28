import torch
import kaolin
import numpy as np
from dmtet_network import Decoder

# 初始化
# path to the point cloud to be reconstructed
pcd_path = "../samples/bear_pointcloud.usd"
# path to the output logs (readable with the training visualizer in the omniverse app)
logs_path = './logs/'

# We initialize the timelapse that will store USD for the visualization apps
timelapse = kaolin.visualize.Timelapse(logs_path)
# arguments and hyperparameters
device = 'cuda'
lr = 1e-3
laplacian_weight = 0.1
iterations = 5000
save_every = 100
multires = 2
grid_res = 128
# ------------------------------------------------------------------------------

# 加载点云，做归一化操作

# 这个point的数据结构是一个数组，数组的每个元素是一个三维数组，代表每个点云的三维坐标
points = kaolin.io.usd.import_pointclouds(pcd_path)[0].points.to(device)
# point.shape()会返回point的形状，有多少行（点云的个数），多少列（在这里一直是3，代表一个点云的三维坐标）
if points.shape[0] > 100000: # 如果点云的个数大于0则执行
    # range()的用法：python中range（）函数的用法是生成一个整数序列，可以用于for循环中1。range（）函数可以接受三个参数，
    # 分别是start，stop和step2。其中，start是可选的，表示序列的起始值，默认为0；stop是必须的，表示序列的终止值，不包含在序列中；
    # step是可选的，表示序列的步长，默认为1

    # list()的用法：python中list（）函数的用法是创建一个列表对象1。列表对象是一种有序且可变的集合，可以存储各种类型的数据1。
    # list（）函数可以接受一个可迭代的参数，例如字符串，元组，集合，字典等，或者不传入任何参数，返回一个空列表

    idx = list(range(points.shape[0])) # 得到点云所含点的个数，是列表数据结构

    # np.random.shuffle（）是一个用于打乱数组或序列内容的函数，它会在原地修改数组或序列，不会返回新的对象1。
    # 这个函数只接受一个参数，即要打乱的数组或序列

    np.random.shuffle(idx) # 打乱idx列表中数字的顺序
    idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long) # 将列表转换成张量，以便投入网络中训练
    points = points[idx] # 这一步是把之前为数组数据格式的点云转换为张量格式的点云

# The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
center = (points.max(0)[0] + points.min(0)[0]) / 2
max_l = (points.max(0)[0] - points.min(0)[0]).max()
points = ((points - center) / max_l)* 0.9
timelapse.add_pointcloud_batch(category='input',
                               pointcloud_list=[points.cpu()], points_type = "usd_geom_points")
# ----------------------------------------------------------------------------------------------------

# Loading the Tetrahedral Grid

# tet_verts是一个torch.tensor对象，它包含了从’…/samples/{}_verts.npz’.format(grid_res)指定的文件中加载的四面体网格的顶点坐标，
# 它的形状是(N, 3)，其中N是顶点的数量，3是坐标的维度
tet_verts = torch.tensor(np.load('../samples/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)

# tets是一个torch.tensor对象，它包含了从’…/samples/{}tets{}.npz’.format(grid_res, i)指定的文件中加载的四面体网格的拓扑结构，
# 它的形状是(M, 4)，其中M是四面体的数量，4是每个四面体由4个顶点组成
tets = torch.tensor(([np.load('../samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]),
                    dtype=torch.long, device=device).permute(1,0)
print (tet_verts.shape, tets.shape)

# 关于其中函数调用的详细解释：
# torch.tensor（）是一个用于创建张量对象的函数，它可以接受一个numpy数组，一个列表，或者一个标量作为参数，并返回一个相应的张量对象。
# 它还可以接受一些可选的参数，例如dtype，device，requires_grad等，来指定张量的数据类型，存储设备，是否需要梯度等属性。

# np.load（）是一个用于加载numpy数组的函数，它可以接受一个文件名或者一个文件对象作为参数，并返回一个numpy数组或者一个字典。
# 如果文件是一个.npz格式的压缩文件，那么返回的是一个字典，其中包含了文件中存储的多个数组。可以通过字典的键来访问对应的数组，
# 例如[‘data’]就是访问键为’data’的数组。

# format（）是一个用于格式化字符串的方法，它可以接受一些参数，并将它们插入到字符串中的占位符{}中。
# 例如’…/samples/{}_verts.npz’.format(grid_res)就是将grid_res这个变量的值替换到{}中，得到一个完整的文件名。

# permute（）是一个用于改变张量维度顺序的方法，它可以接受一些整数作为参数，并按照这些整数指定的顺序重新排列张量的维度。
# 例如permute(1,0)就是将张量的第0维和第1维交换位置。

# kal.rep.TetMesh（）是一个用于定义四面体网格对象的类，它可以接受两个张量作为参数，分别表示四面体网格的顶点坐标和拓扑结构，并返回一个四面体网格对象。
# 这个对象有一些属性和方法，可以用来操作和渲染四面体网格。

# ---------------------------------------------------------------------------------------------------

# Initialize model and create optimizer
model = Decoder(multires=multires).to(device)
model.pre_train_sphere(1000)

# ---------------------------------------------------------------------------------------------------

# Preparing the Losses and Regularizer
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    """
    parameters:
        mesh_verts:mesh_verts是一个张量，表示网格的顶点坐标，它的形状是(V, 3)，其中V是顶点的数量，3是坐标的维度
        mesh_faces:mesh_faces是一个张量，表示网格的面，它的形状是(F, 3)，其中F是面的数量，3是每个面的顶点数（假设都是三角形）

        mesh_faces中的每个元素是一个整数，表示对应的顶点在mesh_verts中的索引。
        例如，如果mesh_faces[0] = [1, 2, 3]，那么第一个面由mesh_verts1, mesh_verts2, mesh_verts3三个顶点组成。

    首先，初始化两个零张量term和norm，它们的形状和mesh_verts相同。
    然后，根据mesh_faces，获取每个三角形面的三个顶点v0, v1, v2。
    接着，使用scatter_add_函数，将每个顶点与其邻居顶点的差累加到term中，将每个顶点的邻居数累加到norm中。
    最后，将term除以norm（避免除以零），得到每个顶点的拉普拉斯坐标

    它的主要思想是计算每个顶点的拉普拉斯坐标，即顶点与其邻居顶点的平均值的差，然后计算这些坐标的平方和，作为正则化项。
    这样做的目的是使网格保持局部平滑性，减少噪声和扭曲。
    """
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


def loss_f(mesh_verts, mesh_faces, points, it):
    """
    这个函数是用来计算网格和点云之间的损失函数的

    parameters
        mesh_verts: 一个张量，表示网格的顶点坐标，它的形状是(V, 3)，其中V是顶点的数量，3是坐标的维度
        mesh_faces: 一个张量，表示网格的面，它的形状是(F, 3)，其中F是面的数量，3是每个面的顶点数（假设都是三角形）
                    mesh_faces中的每个元素是一个整数，表示对应的顶点在mesh_verts中的索引
        points: 一个张量，表示点云，它的形状是(P, 3)，其中P是点的数量，3是坐标的维度
        it: 一个整数，表示迭代次数

    return: 一个标量，表示网格和点云之间的距离损失和正则化损失之和。距离损失是网格表面和点云之间的chamfer distance，
            正则化损失是网格顶点的拉普拉斯正则化项
    """

    # 这段代码是用来从网格上采样50000个点，作为预测的点云。它的输入是网格的顶点和面，它的输出是一个张量，表示采样的点云，它的形状是(50000, 3)。
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]

    # 这段代码是用来计算预测的点云和真实的点云之间的chamfer distance，作为距离损失。
    # 它的输入是两个张量，表示预测的点云和真实的点云，它的输出是一个标量，表示两个点云之间的平均距离。
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()

    if it > iterations//2:
        # 计算拉普拉斯正则项
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        # 返回距离损失和拉普拉斯正则项的总和（拉普拉斯正则项有超参数调制）
        return chamfer + lap * laplacian_weight
    return chamfer

# --------------------------------------------------------------------------------

# Setting up Optimizer
vars = [p for _, p in model.named_parameters()]
optimizer = torch.optim.Adam(vars, lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

# ------------------------------------------------------------------------

# training
for it in range(iterations):
    pred = model(tet_verts) # predict SDF and per-vertex deformation
    sdf, deform = pred[:,0], pred[:,1:]
    verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
    mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra\
        (verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
    mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

    loss = loss_f(mesh_verts, mesh_faces, points, it)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (it) % save_every == 0 or it == (iterations - 1):
        print ('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.
               format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
        # save reconstructed mesh
        timelapse.add_mesh_batch(
            iteration=it+1,
            category='extracted_mesh',
            vertices_list=[mesh_verts.cpu()],
            faces_list=[mesh_faces.cpu()]
        )

