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

# 加载点云，做归一化操作,该点云是真实点云，作为label来训练网络

# 这个point的数据结构是一个数组，数组的大小（行数）是点云的个数，
# 数组的每个元素是一个三维数组，代表每个点云的三维坐标，这里加载了label（熊的点云）
points = kaolin.io.usd.import_pointclouds(pcd_path)[0].points.to(device)

if points.shape[0] > 100000: # 如果点云的个数大于100000则执行

    # 得到点云所含点的个数，是列表数据结构
    idx = list(range(points.shape[0]))

    # 打乱idx列表中数字的顺序，方便之后随机选取100000个点云进行重建
    np.random.shuffle(idx)

    # 只选取点云中的100000个进行重建将列表转换成张量，以便投入网络中训练
    idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)
    points = points[idx] # 这一步是把之前为数组数据格式的点云转换为张量格式的点云

# The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
center = (points.max(0)[0] + points.min(0)[0]) / 2
max_l = (points.max(0)[0] - points.min(0)[0]).max()

# 真实点云，作为label来训练网络
points = ((points - center) / max_l) * 0.9

timelapse.add_pointcloud_batch(category='input',
                               pointcloud_list=[points.cpu()], points_type = "usd_geom_points")
# ----------------------------------------------------------------------------------------------------

# Loading the Tetrahedral Grid

# tet_verts是一个torch.tensor对象，它包含了从’…/samples/{}_verts.npz’.format(grid_res)指定的文件中加载的四面体网格的顶点坐标，
# 它的形状是(V, 3)，其中V是顶点的数量，3是坐标的维度
tet_verts = torch.tensor(np.load('../samples/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)

# tets是一个torch.tensor对象，它包含了从’…/samples/{}tets{}.npz’.format(grid_res, i)指定的文件中加载的四面体网格的拓扑结构，
# 它的形状是(M, 4)，其中M是四面体的数量，4是每个四面体由4个顶点组成
tets = torch.tensor(([np.load('../samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]),
                    dtype=torch.long, device=device).permute(1,0)
print (tet_verts.shape, tets.shape)

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
        mesh_faces:mesh_faces是一个张量，表示网格的面，它的形状是(M, 3)，其中M是面的数量，3是每个面的顶点数（假设都是三角形）

        mesh_faces中的每个元素是一个整数，表示对应的顶点在mesh_verts中的索引。
        例如，如果mesh_faces[0] = [1, 2, 3]，那么第一个面由mesh_verts1, mesh_verts2, mesh_verts3三个顶点组成。

    它的主要思想是计算每个顶点的拉普拉斯坐标，即顶点与其邻居顶点的平均值的差，然后计算这些坐标的平方和，作为正则化项。
    这样做的目的是使网格保持局部平滑性，减少噪声和扭曲。
    """
    # 初始化两个零张量term和norm，它们的形状和mesh_verts相同。
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    # 根据mesh_faces，获取每个三角形面的三个顶点v0, v1, v2。
    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    # 使用scatter_add_函数，将每个顶点与其邻居顶点的差累加到term中，将每个顶点的邻居数累加到norm中。
    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    # 将term除以norm（避免除以零），得到每个顶点的拉普拉斯坐标
    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    # 返回拉普拉斯正则项
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

    # 把初始化的四面体网格投入到模型中，来预测每个顶点的SDF值与每个顶点的位移
    # 这个脚本只是给了如何用DMTet将点云重建为网格，而没有其他的可控制的生成内容
    # 如果要让DMTet根据不同的点云输入做不同3D网格的生成，需要增加一些部分：
    # 1.增加一个模型，根据输入的点云，提取出特征，并将它解码为初始的四面体网格
    # （这个四面体网格包含了该点云的形状信息，四面体的顶点有初始的SDF值，表示该顶点与输入点云的位置关系）
    # 2.之后让DMTet模型根据这个初始四面体网格（包含了输入的点云信息）去重建出网格（通过预测四面体顶点的位移和表面的细分）
    pred = model(tet_verts)
    sdf, deform = pred[:,0], pred[:,1:]
    verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets

    # 通过kaolin提供的API来从SDF值中提取出显示表面，然后把表面的信息数据投入到损失函数中，
    # 在损失函数中，会对模型预测的网格进行点云采样，然后根据采样点云与label计算损失，来进行反向传播更新参数
    mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra\
        (verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0))
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

