import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from time import time
import cv2
from PIL import Image
from pyquaternion import Quaternion
from tensorboardX import SummaryWriter
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from glob import glob
import torch.utils.data
from src.data import SegmentationData
from src.models import compile_model, Up, BevEncode
from src.data import compile_data, worker_rnd_init
from src.tools import SimpleLoss, get_batch_iou, get_val_info, get_rot, cumsum_trick, QuickCumsum
import pandas as pd
from nuscenes.map_expansion.map_api import locations
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box

matplotlib.use('TkAgg')

def same_seeds(seed):
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #ramdom.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_3d(data, grid=True, axis=True, invert_x=False, invert_y=False, y_range=None, x_range=None):
    data = data.view(-1, 3)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if not grid:
        ax.grid(None)
    if not axis:
        ax.axis('off')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if y_range:
        assert isinstance(y_range, list) and len(y_range) == 2 and y_range[0] <= y_range[1]
        ax.set_ylim(ymin=y_range[0], ymax=y_range[1])
        keep = (data[:, 1] < y_range[1]) & (data[:, 0] > x_range[0])
        data = data[keep]
    if x_range:
        assert isinstance(x_range, list) and len(x_range) == 2 and x_range[0] <= x_range[1]
        ax.set_ylim(ymin=x_range[0], ymax=x_range[1])
        keep = (data[:, 1] < x_range[1]) & (data[:, 0] > x_range[0])
        data = data[keep]
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


same_seeds(0)
#----------------------------------------------------config start--------------------------------------------------------
dataroot= '/home/mengmeng/data/nuScenes'
nepochs = 10000
gpuid = 0

H, W = 900, 1600
resize_lim = (0.193, 0.225)
final_dim = (128, 352)
bot_pct_lim =(0.0, 0.22)   # bottom crop percent
rot_lim = (-5.4, 5.4)  # training rotation img range
rand_flip = True
ncams = 5
max_grad_norm = 5.0
pos_weight = 2.13  # loss positive weight params
logdir = './runs'

xbound = [-50.0, 50.0, 0.5]
ybound = [-50.0, 50.0, 0.5]
zbound = [-10.0, 10.0, 20.0]
dbound = [4.0, 45.0, 1]   # depth

bsz = 2   # batchsize
nworkers = 0   # threads
lr = 1e-3
weight_decay = 1e-7

grid_conf = {
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
}

data_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': ncams
}

# device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
device = torch.device('cpu')
#-------------------------------------------config end------------------------------------------------------------------


#-------------------------------------------SegmentationData Start------------------------------------------------------
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)  #load dataset
traindata = SegmentationData(nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf)


#-------------------------------------------prepro start----------------------------------------------------------------
is_train = True
split = {
    'v1.0-trainval': {True: 'train', False: 'val'},
    'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
}[nusc.version][is_train]

scenes = create_splits_scenes()[split]
samples = [samp for samp in nusc.sample]
# remove samples that aren't in this split
samples = [samp for samp in samples if
           nusc.get('scene', samp['scene_token'])['name'] in scenes]

# sort by scene, timestamp (only to make chronological viz easier)
samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

index = 100
rec = samples[index]
#--------------------------------------------prepro end-----------------------------------------------------------------

#-------------------------------------------choose_cams start-----------------------------------------------------------
if is_train and data_aug_conf['Ncams'] < len(data_aug_conf['cams']):
    cams = np.random.choice(data_aug_conf['cams'], data_aug_conf['Ncams'], replace=False)
else:
    cams = data_aug_conf['cams']
# print(cams)
#--------------------------------------------choose_cams end------------------------------------------------------------

#---------------------------------------------get_image_data start------------------------------------------------------
# cam = next(iter(cams))
cam = 'CAM_FRONT'
samp = nusc.get('sample_data', rec['data'][cam])
imgname = os.path.join(nusc.dataroot, samp['filename'])
img = Image.open(imgname)
post_rot = torch.eye(2) # aug before and after Affine Transformation Matrix
post_tran = torch.zeros(2)  # aug before and after Affine Transformation Matrix

sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
intrin = torch.Tensor(sens['camera_intrinsic'])
rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # camera to ego
tran = torch.Tensor(sens['translation'])  # camear to ego

# map expension
egopose = nusc.get('ego_pose', samp['ego_pose_token'])
ego_rot = torch.tensor(Quaternion(egopose['rotation']).rotation_matrix)
ego_tran = torch.tensor(egopose['translation'])

#--------------------------------------------------------sample_augmentation start--------------------------------------
H, W = data_aug_conf['H'], data_aug_conf['W']  # H: 900, W: 1600
fH, fW = data_aug_conf['final_dim']  # (128, 352) # final size
if is_train:
    resize = np.random.uniform(*data_aug_conf['resize_lim'])
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim'])) * newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
        flip = True
    rotate = np.random.uniform(*data_aug_conf['rot_lim'])
# else: #test data aug
#     resize = max(fH / H, fW / W)
#     resize_dims = (int(W * resize), int(H * resize))
#     newW, newH = resize_dims
#     crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
#     crop_w = int(max(0, newW - fW) / 2)
#     crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
#     flip = False
#     rotate = 0
# adjust image
img = img.resize(resize_dims)  # scaling
img = img.crop(crop)
if flip:
    img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.rotate(rotate)
# img.show()
# img.save('img_aug.png')
# post-homography transformation
# 数据增强后的某一点坐标需要对应回增强前的位置,
post_rot *= resize
post_tran -= torch.Tensor(crop[:2])
if flip:
    A = torch.Tensor([[-1, 0], [0, 1]])
    b = torch.Tensor([crop[2] - crop[0], 0])
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
A = get_rot(rotate/180*np.pi)  # obtained aug rotation operation's rotation matrix
b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2 # reserve part image and center coordinates (176, 64)
b = A.matmul(-b) + b
post_rot = A.matmul(post_rot)
post_tran = A.matmul(post_tran) + b
#---------------------------------------------------img_transform end---------------------------------------------------
# for convenience, make augmentation matrices 3x3
post_tran_2 = torch.zeros(3)
post_rot_2 = torch.eye(3)
post_tran_2[:2] = post_tran
post_rot_2[:2, :2] = post_rot
#---------------------------------------------get_image_data end--------------------------------------------------------
'''

#---------------------------------------------get_binimg start----------------------------------------------------------
dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # x, y, z directions grid size
bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])  # x, y, z directions the first gtid coordinates
nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])  # x, y, z directions grid numbers
dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()

egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
trans = -np.array(egopose['translation'])
rot = Quaternion(egopose['rotation']).inverse
binimg = np.zeros((nx[0], nx[1]))

patch_h = ybound[1] - ybound[0]
patch_w = xbound[1] - xbound[0]
map_pose = np.array(egopose['translation'])[:2]
patch_box = (map_pose[0], map_pose[1], patch_h, patch_w)
canvas_h = int(patch_h / ybound[2])
canvas_w = int(patch_w / xbound[2])
canvas_size = (canvas_h, canvas_w)

rotation = Quaternion(egopose['rotation']).rotation_matrix
v = np.dot(rotation, np.array([1, 0, 0]))
yaw = np.arctan2(v[1], v[0])   # np.arctan2(y, x), positive clock
patch_angle = yaw / np.pi * 180

layer_names = ['drivable_area']
maps = {}
for location in locations:
    maps[location] = NuScenesMap(dataroot, location)

location = nusc.get('log', nusc.get('scene', rec['scene_token'])['log_token'])['location']
masks = maps[location].get_map_mask(patch_box=patch_box,
                                    patch_angle=patch_angle,
                                    layer_names=layer_names,
                                    canvas_size=canvas_size)

masks = masks.transpose(0, 2, 1)
masks = masks.astype(np.bool)
binimg[masks[0]] = 0.5
cv2.imshow('binimg', binimg)
key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()

for tok in rec['anns']: # iterate each annotation token of this sample
    inst = nusc.get('sample_annotation', tok) # find this annotation
    # add category for lyft
    if not inst['category_name'].split('.')[0] == 'vehicle':  # only vehicle
        continue
    box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))  # center, size, orientation
    box.translate(trans)  # translate the center coordinates of the box from the global to ego.
    box.rotate(rot)  # translate the center coordinates of the box from the global to ego.

    pts = box.bottom_corners()[:2].T # bottom four corners
    pts = np.round((pts - bx[:2] + dx[:2] / 2.) / dx[:2]).astype(np.int32)  #translate bbox to grid coordinates and move range from [-50, 50] to [0, 100]
    pts[:, [1, 0]] = pts[:, [0, 1]]  # change (x, y) to (y, x)
    cv2.fillPoly(binimg, [pts], 1.0)
# cv2.imshow('binimg', binimg)
# key = cv2.waitKey(0)
# cv2.imshow('seg', masks.sequeeze(0)*255)
import matplotlib.image as mpimg
image_paths = []
for cam in ['CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_BACK']:
    samp = nusc.get('sample_data', rec['data'][cam])
    image_path = os.path.join(nusc.dataroot, samp['filename'])
    image_paths.append(image_path)
# 2 * 3 canvas
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axes.flat):
    if i < len(image_paths):
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.set_title(f"image {i + 1}")
        ax.axis("off")  # hide axis
plt.tight_layout()
plt.show()
#---------------------------------------------get_binimg end------------------------------------------------------------
#---------------------------------------------Segmentation end----------------------------------------------------------
'''
#---------------------------------------------load data start-----------------------------------------------------------
trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                          shuffle=True,
                                          num_workers=nworkers,
                                          drop_last=True,
                                          worker_init_fn=worker_rnd_init)
imgs, rots, trans, intrins, post_rots, post_trans, binimgs = next(iter(trainloader))
imgs = imgs.to(device)
rots = rots.to(device)
trans = trans.to(device)
intrins = intrins.to(device)
post_rots = post_rots.to(device)
post_trans = post_trans.to(device)

#---------------------------------------------load data end-------------------------------------------------------------

#---------------------------------------------model start---------------------------------------------------------------
outC = 1
downsample = 16
# {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1]}
dx = torch.tensor([row[2] for row in [xbound, ybound, zbound]], device=device)
bx = torch.tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]], device=device) # the first point coordinates
nx = torch.tensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]], dtype=torch.long, device=device) # grid numbers
dx = nn.Parameter(dx, requires_grad=False)  # [0.5, 0.5, 20]
bx = nn.Parameter(bx, requires_grad=False)  # [-49.7500, -49.7500, 0.0000]
nx = nn.Parameter(nx, requires_grad=False)  # [200, 200, 1]

# make grid in image plane
ogfH, ogfW = data_aug_conf['final_dim']  # after date preprocessing ofgH: 128, ofgW: 352 <-- original: 900, 1600
fH, fW = ogfH // downsample, ogfW // downsample  # after downsampling, ofgH: 8, ofgW: 22
ds = torch.arange(*grid_conf['dbound'], dtype=torch.float, device=device).view(-1, 1, 1).expand(-1, fH, fW)  # ds.shape: torch.Size([41, 8, 22])
D, _, _ = ds.shape # D: 41 means depth direction grid numbers
xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float, device=device).view(-1, 1, fW).expand(D, fH, fW)  # torch.Size([41, 8, 22]), feature map point corresponding to ogfHW location
ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float, device=device).view(-1, fH, 1).expand(D, fH, fW)  # torch.Size([41, 8, 22])

frustum = torch.stack((xs, ys, ds), -1)   # torch.Size([41, 8, 22, 3]) D * H * W * 3
frustum = nn.Parameter(frustum, requires_grad=False)

#---------------------------------------------CamEncode init start------------------------------------------------------
camC = 64
CamEncode_trunk = EfficientNet.from_pretrained("efficientnet-b0").to(device)
CamEncode_up1 = Up((320 + 112), 512).to(device)
CamEncode_depthnet = nn.Conv2d(512, D + camC, kernel_size=1, padding=0).to(device)
#---------------------------------------------CamEncode init end--------------------------------------------------------
bevencode = BevEncode(inC=camC, outC=outC).to(device)
use_quickcumsum = True
#-------------------------------------------------Model init end--------------------------------------------------------
#-------------------------------------------------Model forward start---------------------------------------------------
# def forward(self, x, rots, trans, intrins, post_rots, post_trans):
#     x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
#     x = self.bevencode(x)
#     return x
# x: imgs.shape: torch.Size([2, 5, 3, 128, 352]) [B, N, C, H, W]
# rots.shape: torch.Size([2, 5, 3, 3))
# trans.shape: torch.Size([2, 5, 3])
# intrins.shape: torch.Size([2, 5, 3, 3])
# post_rots.shape: torch.Size([2, 5, 3, 3])
# post_trans.shape: torch.Size([2, 5, 3])
#-------------------------------------------------Model forward start---------------------------------------------------
#------------------------------------------------get geometry start-----------------------------------------------------
B, N, _ = trans.shape  # B:2 N: 5

# undo post-transformation
# B x N x D x H x W x 3
# frustum.shape: torch.Size([41, 8, 22, 3])
# post_trans.shape: torch.Size([2, 5, 3])
points = frustum - post_trans.view(B, N, 1, 1, 1, 3)  # points.shape: torch.Size([2, 5, 41, 8, 22, 3]) batch: 2 camera: 5, depth: 41
points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))  # points.shape:
# The code maps the coordinates of the image after data augmentation to the coordinates of the original image.

# cam_to_ego
points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],points[:, :, :, :, :, 2:3]), 5)  # point.shape:torch.Size([2, 5, 41, 8, 22, 3, 1])
# (u, v, d) -> (ud, vd, d) Homogeneous coordinates
# rots: 2D -> 3D
combine = rots.matmul(torch.inverse(intrins))  # combine.shape:  torch.Size([2, 5, 3, 3])
points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # points.shape:torch.Size([2, 5, 41, 8, 22, 3])
points += trans.view(B, N, 1, 1, 1, 3)
# pixel coordinates[u, v, 1] -> ego coordinates[x, y, z]
# plot_3d(points[0, 2, ...].cpu())
#------------------------------------------------get_geometry end-------------------------------------------------------
#------------------------------------------------get_cam_fests start----------------------------------------------------
x = imgs  # x: torch.Size([2, 5, 3, 128, 352])
B, N, C, imH, imW = x.shape
x = x.view(B * N, C, imH, imW)  # x.shape:torch.Size([10, 3, 128, 352]) -->  [B*N, C, H, W]
#------------------------------------------------CamEncode forward start------------------------------------------------
#------------------------------------------------get_depth_feat start----------------------------------------------------
#------------------------------------------------get_eff_depth start----------------------------------------------------
# def get_eff_depth(self, x):
# adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
endpoints = dict()

# Stem
x = CamEncode_trunk._swish(CamEncode_trunk._bn0(CamEncode_trunk._conv_stem(x)))  # x: 10 x 32 x 64 x 176
prev_x = x

# Blocks
for idx, block in enumerate(CamEncode_trunk._blocks):
    drop_connect_rate = CamEncode_trunk._global_params.drop_connect_rate
    if drop_connect_rate:
        drop_connect_rate *= float(idx) / len(CamEncode_trunk._blocks)  # scale drop connect_rate
    x = block(x, drop_connect_rate=drop_connect_rate)
    if prev_x.size(2) > x.size(2):
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
    prev_x = x

# Head
endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
# reduction_1 torch.Size([10, 16, 64, 176])
# reduction_2 torch.Size([10, 24, 32, 88])
# reduction_3 torch.Size([10, 24, 16, 44])
# reduction_4 torch.Size([10, 24, 8, 22])
# reduction_5 torch.Size([10, 24, 4, 11])
x = CamEncode_up1(endpoints['reduction_5'], endpoints['reduction_4'])  # x.shape: torch.Size([10, 512, 8, 22])
#------------------------------------------------get_eff_depth end------------------------------------------------------
# CamEncode_depthnet  = nn.Conv2d(512, D + C, kernel_size=1, padding=0)
# camC = 64  # img feature dim
# D = 41  # depth direction grid numbers
x = CamEncode_depthnet(x)  # x.shape: torch.Size([10, 105, 8, 22])  512 --> 105 == camC + D
depth = x[:, :D].softmax(dim=1)  # depth.shape: torch.Size([10, 41, 8, 22]] first 41 from 105
x = depth.unsqueeze(1) * x[:, D:(D + camC)].unsqueeze(2)  # first add dimension on idx 1 of depth
# depth.unsqueenze(1).shape: torch.Size([10, 1, 41, 8, 22]) depth dim
#  x[:, D:(D + camC)].unsqueeze(2).shape: torch.Size([10, 64, 1, 8, 22]) img feature
# x.shape: torch.Size([10, 64, 41, 8, 22])
#------------------------------------------------CamEncode forword end--------------------------------------------------
x = x.view(B, N, camC, D, imH // downsample, imW // downsample)  # x.shape: torch.Size([2, 5, 64, 41, 8, 22])
x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, H, W, C] -> torch.Size([2, 5, 41, 8, 22, 64])
#------------------------------------------------get_cam_fests end------------------------------------------------------
#------------------------------------------------voxel_pooling start----------------------------------------------------
geom_feats = points  # torch.Size([2, 5, 41, 8, 22, 3])
B, N, D, H, W, C = x.shape
Nprime = B*N*D*H*W  # 72160

# flatten x
x = x.reshape(Nprime, C)  # torch.Size([72160, 64])

# flatten indices
geom_feats = ((geom_feats - (bx - dx/2.)) / dx).long()
geom_feats = geom_feats.view(Nprime, 3)  # torch.Size([72160, 3])
batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                     device=x.device, dtype=torch.long) for ix in range(B)])
# batch_ix.shape # torch.Size([72160, 1])
geom_feats = torch.cat((geom_feats, batch_ix), 1)
# geom_feats.shape  # torch.Size([72160, 4])

# filter out points that are outside box. 因为41 份depth是人为划分出来的，不是每个depth都是实际的意义
# print('nx:', nx) # nx: tensor([200, 200, 1]) x, y, z three direction grid numbers
# 过滤掉在边界线之外的点 x:0~199， y:0~199, z:0
kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0])\
    & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1])\
    & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
x = x[kept]  # torch.Size([69001, 64])
geom_feats = geom_feats[kept] # torch.Size([69001, 4]) (4: x, y, z, b)

# get tensors from the same voxel next to each other
# each grid of each batch have only one rank value
ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B)\
    + geom_feats[:, 1] * (nx[2] * B)\
    + geom_feats[:, 2] * B\
    + geom_feats[:, 3]
# rank 和 each voxel 是一一对应的
sorts = ranks.argsort()  #shape: 69001
x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 通过排序将Voxel索引相近的排在一起，更重要的是把在同一个Voxel里面的排在一起
# x.shape torch.Size([69001, 64])
# geom_feats.shape torch.Size([69001, 4])
# ranks.shape torch.Size([69001])

# cumsum trick
x = x.cumsum(0)  # 累加
kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)  # x.shape[0]: 690001
kept[:-1] = (ranks[1:] != ranks[:-1])

x, geom_feats = x[kept], geom_feats[kept]  # x.shape: torch.Size([22121, 64]); geom_feats.shape:torch.Size([22121, 4])
x = torch.cat((x[:1], x[1:] - x[:-1]))

# griddify (B x C x Z x X x Y)
final = torch.zeros((B, C, nx[2], nx[0], nx[1]), device=x.device)  # torch.Size([2, 64, 1, 200, 200])
final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  #最后将x按照grid坐标放到final中, geom_feats代表3d空间Voxel 的索引值

# collapse Z
final = torch.cat(final.unbind(dim=2), 1)  # torch.Size([2, 64, 200, 200]) # splat operation!!!

#------------------------------------------------voxel_pooling end------------------------------------------------------
x = bevencode(final)  # x.shape torch.Size([2, 1, 200, 200])
#------------------------------------------------model forward end------------------------------------------------------
#-------------------------------------------------model end-------------------------------------------------------------


#-------------------------------------------------SimpleLoss start------------------------------------------------------
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
loss_fn = loss_fn(x, binimgs) # binimgs.shape: torch.Size([2, 1, 200, 200])

#-------------------------------------------------SimpleLoss end------------------------------------------------------
print('GOOD!')
