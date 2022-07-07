import config
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch.autograd import Variable
from torch import ones, zeros, randn
from torchnet.meter import AverageValueMeter
from model import NetG, NetD
from train import train

if config.vis:
    from visualize import Visualizer

    vis = Visualizer(config.env)

tf = transforms.Compose(
    [transforms.Resize(config.image_size), transforms.CenterCrop(config.image_size), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ds = datasets.ImageFolder(config.data_path, transform=tf)
dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
net_g, net_d = NetG(config), NetD(config)
opt_g, opt_d = Adam(net_g.parameters(), config.lr1, betas=(config.beta1, 0.999)), Adam(net_d.parameters(), config.lr2,
                                                                                       betas=(config.beta1, 0.999))
loss = BCELoss()
true_labels = Variable(ones(config.batch_size))
fake_labels = Variable(zeros(config.batch_size))
fix_noises = Variable(randn(config.batch_size, config.nz, 1, 1))
noises = Variable(randn(config.batch_size, config.nz, 1, 1))
meter_d = AverageValueMeter()
meter_g = AverageValueMeter()

train(dl, net_g, net_d, opt_g, opt_d, meter_g, meter_d, loss, true_labels, fake_labels, noises, fix_noises,
      config.max_epoch, vis)
