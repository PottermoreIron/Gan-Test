from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision
import config
import os
import ipdb


def train(data, net_g, net_d, optimizer_g, optimizer_d, meter_g, meter_d, loss, true_labels, fake_labels, noises,
          fix_noises, epoch, vis):
    for e in range(epoch):
        for i, (img, _) in tqdm(enumerate(data)):
            real_img = Variable(img)
            if i % config.d_every == 0:
                optimizer_d.zero_grad()
                out = net_d(real_img)
                err_d_real = loss(out, true_labels)
                err_d_real.backward()
                noises.data.copy_(torch.randn(config.batch_size, config.nz, 1, 1))
                fake_img = net_g(noises).detach()
                out = net_d(fake_img)
                err_d_fake = loss(out, fake_labels)
                err_d_fake.backward()
                optimizer_d.step()
                err_d = err_d_fake + err_d_real
                meter_d.add(err_d.item())
            if i % config.g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(config.batch_size, config.nz, 1, 1))
                fake_img = net_g(noises)
                out = net_d(fake_img)
                err_g = loss(out, true_labels)
                err_g.backward()
                optimizer_g.step()
                meter_g.add(err_g.item())
            if config.vis and i % config.plot_every == config.plot_every - 1:
                if os.path.exists(config.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = net_g(fix_noises)
                vis.images(fix_fake_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('d', meter_d.value()[0])
                vis.plot('g', meter_g.value()[0])
        if epoch % config.decay_every == 0:
            # 保存模型、图片
            torchvision.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (config.save_path, epoch),
                                         normalize=True,
                                         range=(-1, 1))
            torch.save(net_d.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            torch.save(net_g.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            meter_d.reset()
            meter_g.reset()
            optimizer_g = torch.optim.Adam(net_g.parameters(), config.lr1, betas=(config.beta1, 0.999))
            optimizer_d = torch.optim.Adam(net_d.parameters(), config.lr2, betas=(config.beta1, 0.999))
