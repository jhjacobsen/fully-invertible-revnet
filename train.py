import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import math
import numpy as np
import pdb
import argparse
import time

from invertible_layers import * 
from utils import * 

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--depth', type=int, default=16) 
parser.add_argument('--n_levels', type=int, default=3) 
parser.add_argument('--norm', type=str, default='actnorm')
parser.add_argument('--permutation', type=str, default='reverse')
parser.add_argument('--coupling', type=str, default='additive')
parser.add_argument('--n_bits_x', type=int, default=8.)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--learntop', action='store_true')
parser.add_argument('--n_warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=1e-3)
# logging
parser.add_argument('--print_every', type=int, default=500, help='print NLL every _ minibatches')
parser.add_argument('--test_every', type=int, default=5, help='test on valid every _ epochs')
parser.add_argument('--save_every', type=int, default=5, help='save model every _ epochs')
parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--save_dir', type=str, default='exps', help='directory for log / saving')
parser.add_argument('--load_dir', type=str, default=None, help='directory from which to load existing model')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# loading / dataset preprocessing
tf_test = transforms.Compose([transforms.Pad(2, padding_mode="constant"),
                         transforms.ToTensor(),
                         lambda x: torch.cat((torch.zeros_like(x),torch.zeros_like(x),x),0)])

tf = transforms.Compose([transforms.Pad(2, padding_mode="constant"),
                         transforms.RandomCrop(32),
                         transforms.ToTensor(),
                         lambda x: torch.cat((torch.zeros_like(x),torch.zeros_like(x),x),0), 
                         lambda x: x + torch.zeros_like(x).uniform_(0., 1./args.n_bins)])

#train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=True, 
    download=True, transform=tf), batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)

#test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
    transform=tf_test), batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True)

# construct model and ship to GPU
model = Glow_((args.batch_size, 3, 32, 32), args).cuda()
#print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# set up the optimizer
params = list(model.parameters())
bn = nn.BatchNorm1d(10,momentum=0.9).cuda()
params += list(bn.parameters())
ac = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 10),
          )
ac = ac.cuda()
optim_main = optim.Adamax(params, lr=1e-3)
optim_ac = optim.Adamax(ac.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)

# data dependant init
init_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=True, 
    download=True, transform=tf), batch_size=512, shuffle=True, num_workers=1)

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

with torch.no_grad():
    model.eval()
    for (img, _) in init_loader:
        img = img.cuda()
        img *= 2. 
        img -= 1.
        objective = torch.zeros_like(img[:, 0, 0, 0])
        zn = []
        _ = model(img, objective, zn)
        break

# once init is done, we leverage Data Parallel
model = nn.DataParallel(model).cuda()
start_epoch = 0
ce_loss = nn.CrossEntropyLoss()

# load trained model if necessary (must be done after DataParallel)
if args.load_dir is not None: 
    model, optim, start_epoch = load_session(model, optim, args)

# training loop
# ------------------------------------------------------------------------------
for epoch in range(start_epoch, args.n_epochs):
    print('epoch %s' % epoch)
    model.train()
    ac.train()
    bn.train()
    avg_train_bits_x = 0.
    num_batches = len(train_loader)
    lr = learning_rate(args.lr, epoch)
    for param_group in optim_main.param_groups:
        param_group['lr'] = lr
    for i, (img, label) in enumerate(train_loader):
        # if i > 10 : break
        
        t = time.time()
        img = img.cuda() 
        img *= 2.
        img -= 1.
        label = label.cuda() 
        label_ac = label[torch.randperm(label.size()[0])] # torch.ones_like(label)*0.1
        #label_ac = torch.ones_like(label)*0.1 # label[torch.randperm(label.size()[0])] # torch.ones_like(label)*0.1
        objective = torch.zeros_like(img[:, 0, 0, 0])

        # discretizing cost 
        objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
        
        # log_det_jacobian cost (and some prior from Split OP) 
        zn = []
        z, objective, zn = model(img, objective, zn)
        logits = z[:,:10,2,2]
        logits = bn(logits)
        _zn = z.clone()
        _zn[:,:10,2,2] = 0.1
        _zn = torch.cat((_zn.view(_zn.size(0),-1),zn[0].view(zn[0].size(0),-1),zn[1].view(zn[1].size(0),-1)),dim=1) 

        #ac.train()
        for k in range(5):
          logits_ac = ac(_zn.detach())
          pyx_ac_loss = ce_loss(logits_ac, label)
          ac_obj = torch.mean(pyx_ac_loss)
          optim_ac.zero_grad()
          ac_obj.backward()
          optim_ac.step()
        #ac.eval()

        logits_ac = ac(_zn)
        pyx_loss = ce_loss(logits,label)
        pyx_ac_loss = ce_loss(logits_ac,label_ac)

        nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))+ pyx_loss + 0.05*pyx_ac_loss
        #nll = pyx_loss - 0.1*pyx_ac_loss
        
        # Generative loss
        obj = torch.mean(nll)
        
        optim_main.zero_grad()
        obj.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 200)
        optim_main.step()
        avg_train_bits_x += obj.item()

        # update learning rate
        new_lr = float(args.lr * min(1., (i + epoch * num_batches) / (args.n_warmup * num_batches)))
        #for pg in optim.param_groups: pg['lr'] = learning_rate(new_lr, epoch)
        #for pg in optim.param_groups: pg['lr'] = new_lr

        if (i + 1) % args.print_every == 0: 
            acc = torch.eq(logits.argmax(dim=1),label).sum().double()/len(label)
            acc_ac = torch.eq(logits_ac.argmax(dim=1),label).sum().double()/len(label)
            print('avg train bits per pixel {:.4f}'.format(avg_train_bits_x / args.print_every))
            print('train accuracy {:.4f} - train ac accuracy {:.4f} - train ce loss {:.4f} - train ac ce loss {:.4f}'.format(acc.item(), acc_ac.item(), pyx_loss.mean().item(), pyx_ac_loss.mean().item()))
            avg_train_bits_x = 0.
            model.eval()
            zn = []
            z, objective, zn = model(img, objective, zn)
            zn_met = zn.copy()
            sample = model.module.sample(z,zn)
            zs = z[:,:10,2,2]
            perm = torch.randperm(zs.size()[0])
            zs=zs[perm]
            z[:,:10,2,2] = zs
            sample_met = model.module.sample(z,zn_met)
            sample_perm = sample[perm]
            sample = torch.cat((img[:20],sample[:20],sample_met[:20],sample_perm[:20]),dim=0)
            grid = utils.make_grid(sample,nrow=20)
            utils.save_image(grid, './samples/cifar_Test_{}_{}.png'.format(epoch, i // args.print_every))
            model.train()

        #print('iteration took {:.4f}'.format(time.time() - t))
        
    # test loop
    # --------------------------------------------------------------------------
    acc = 0.
    
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        bn.eval()
        ac.eval()
        avg_test_bits_x = 0.
        with torch.no_grad():
            for i, (img, label) in enumerate(test_loader): 
                # if i > 10 : break
                img = img.cuda() 
                img *= 2.
                img -= 1.
                label = label.cuda() 
                objective = torch.zeros_like(img[:, 0, 0, 0])
               
                # discretizing cost 
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
                
                # log_det_jacobian cost (and some prior from Split OP)
                zn = []
                z, objective, zn = model(img, objective, zn)
                logits = z[:,:10,2,2]
                #logits = lin(logits)
                logits = bn(logits)

                anti_logits = z[:,:10,2,2]
                acc += torch.eq(logits.argmax(dim=1),label).sum().double()
                last_img = img

                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))
                
                # Generative loss
                nobj = torch.mean(nll)
                avg_test_bits_x += nobj
            acc/=10000
            print('avg test bits per pixel {:.4f}'.format(avg_test_bits_x.item() / i))
            print('test accuracy {:.4f} '.format(acc.item()))

            zn = []

            z, objective, zn = model(img, objective, zn)
            zn_met = zn.copy()
            zs = z[:,:10,2,2]
            zs=zs[torch.randperm(zs.size()[0])]
            z[:,:10,2,2] = zs
            sample_met = model.module.sample(z, zn_met)
            grid_met = utils.make_grid(sample_met)
            utils.save_image(grid_met, './samples/cifar_met_Test_{}.png'.format(epoch))

            # reconstruct
            x_hat = model.module.reverse_(z, objective, zn)[0]
            grid = utils.make_grid(x_hat)
            utils.save_image(grid, './samples/cifar_Test_Recon{}.png'.format(epoch))
        
            grid = utils.make_grid(last_img)
            utils.save_image(grid, './samples/cifar_Test_Target.png')


    if (epoch + 1) % args.save_every == 0: 
        save_session(model, optim_main, args, epoch)

