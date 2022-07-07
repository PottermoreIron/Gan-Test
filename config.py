import torch
data_path = 'data'
# 多进程加载数据所用的进程数
num_workers = 0
# 图片尺寸
image_size = 96
batch_size = 256
max_epoch = 200
# 生成器的学习率
lr1 = 2e-4
# 判别器的学习率
lr2 = 2e-4
# Adam优化器的beta1参数
beta1 = 0.5
# 是否使用GPU
gpu = True
# 噪声维度
nz = 100
# 生成器feature map数
ngf = 64
# 判别器feature map数
ndf = 64
# 生成图片保存路径
save_path = 'imgs/'
# 是否使用visdom可视化
vis = True
# visdom的env
env = 'GAN'
# 每间隔20 batch，visdom画图一次
plot_every = 20
# 存在该文件则进入debug模式
debug_file = '/tmp/debuggan'
# 每1个batch训练一次判别器
d_every = 1
# 每5个batch训练一次生成器
g_every = 5
# 没10个epoch保存一次模型
decay_every = 10
# 'checkpoints/netd_.pth' #预训练模型
netd_path = None
# 'checkpoints/netg_211.pth'
netg_path = None

# 只测试不训练
gen_img = 'result.png'
# 从512张生成的图片中保存最好的64张
gen_num = 64
gen_search_num = 512
# 噪声的均值
gen_mean = 0
# 噪声的方差
gen_std = 1
gpu = '' if not torch.cuda.is_available() else 0
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")