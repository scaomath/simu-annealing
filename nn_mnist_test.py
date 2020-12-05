#%%
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary

from tqdm import tqdm
# import Simulated annealing optimizer
from sa import UniformSampler, GaussianSampler
#%%
args = {'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 10,
        'lr': 0.01,
        'momentum': 0.9,
        'no_cuda': False,
        'seed': 1,
        'log_interval': 10}


#%%
args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])

#%%
kwargs = {'num_workers': 1, 'pin_memory': True} if args['cuda'] else {}
lib_path = "/home/scao/Documents/simu-annealing/"
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(lib_path+'data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(lib_path+'data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def conv_block(in_channels, out_channels, stride=1, padding=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=padding, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 stride=1, padding=1, downsample=None, debug=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_block(in_channels, out_channels, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(out_channels, out_channels, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.debug = debug

    def forward(self, x):
        residual = x
        if self.debug: print(f"\n     Original shape: {x.shape}")

        out = self.conv1(x)
        if self.debug: print(f"\nConv2d #1 out shape: {out.shape}")

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.debug: print(f"\nConv2d #2 out shape: {out.shape}")

        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
            if self.debug: print(f"\n   Downsampled shape: {residual.shape}")

        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, debug=False):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv_block(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], 1)
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        self.debug = debug

    def make_layer(self, block, out_channels, blocks, stride=1, padding=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv_block(self.in_channels, out_channels, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, padding, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        if self.debug: print(f"\nLayer 1 out shape: {out.shape}")
       
        out = self.layer2(out)
        if self.debug: print(f"\nLayer 2 out shape: {out.shape}")

        out = self.layer3(out)
        if self.debug: print(f"\nLayer 2 out shape: {out.shape}")

        out = self.avg_pool(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out)

class SimulatedAnnealing(Optimizer):
    def __init__(self, params, sampler, tau0=1.0, anneal_rate=0.0003,
                 min_temp=1e-5, anneal_every=100000, hard=False, hard_rate=0.9, decay_rate=0.9):
        defaults = dict(sampler=sampler, tau0=tau0, tau=tau0,           anneal_rate=anneal_rate,
                        min_temp=min_temp, anneal_every=anneal_every,
                        hard=hard, hard_rate=hard_rate, 
                        decay_rate = decay_rate,
                        iteration=0)
        super(SimulatedAnnealing, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise Exception("loss closure is required to do SA")

        loss = closure()

        for group in self.param_groups:
            # the sampler samples randomness
            # that is used in optimizations
            sampler = group['sampler']

            # clone all of the params to keep in case we need to swap back
            cloned_params = [p.clone() for p in group['params']]

            for p in group['params']:
                # anneal tau if it matches the requirements
                if group['iteration'] > 0 \
                   and group['iteration'] % group['anneal_every'] == 0:
                    if not group['hard']:
                        # smoother annealing: consider using this over hard annealing
                        rate = -group['anneal_rate'] * group['iteration']
                        group['tau'] = np.maximum(group['tau0'] * np.exp(rate),
                                                  group['min_temp'])
                    else:
                        # hard annealing
                        group['tau'] = np.maximum(group['hard_rate'] * group['tau'],
                                                  group['min_temp'])
                decay_rate = np.exp(-group['decay_rate'])

                random_perturbation = decay_rate*group['sampler'].sample(p.data.size())
                p.data = p.data / torch.norm(p.data)
                p.data.add_(random_perturbation)
                group['iteration'] += 1

            # re-evaluate the loss function with the perturbed params
            # if we didn't accept the new params swap back and return
            loss_perturbed = closure(weight=self.param_groups)

            final_loss, is_swapped = self.anneal(loss, loss_perturbed, group['tau'])
            if is_swapped:
                for p, pbkp in zip(group['params'], cloned_params):
                    p.data = pbkp.data

            return final_loss


    def anneal(self, loss, loss_perturbed, tau):
        '''returns loss, is_new_loss'''
        def acceptance_prob(old, new, temp):
            return torch.exp((old - new)/temp)

        # print(loss_perturbed.shape)

        if loss_perturbed.item() < loss.item():
            return loss_perturbed, True
        else:
            # evaluate the metropolis criterion
            ap = acceptance_prob(loss, loss_perturbed, tau)
            # print("old = ", loss.item(), "| pert = ", loss_perturbed.item(),
            #       " | ap = ", ap.item(), " | tau = ", tau)
            if ap.item() > np.random.rand():
                return loss_perturbed, True

            # return the original loss if above fails
            # or if the temp is now annealed
            return loss, False

# %%
if __name__ == "__main__":
    

    args = {'batch_size': 64,
            'test_batch_size': 1000,
            'epochs': 10,
            'lr': 0.01,
            'momentum': 0.9,
            'no_cuda': False,
            'seed': 1,
            'log_interval': 10,
            'sgd': True,
            'resnet': True}
    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()
    
    model = Net()

    if args['resnet']:
        net_args = { "block": ResidualBlock,
                    "layers": [2, 2, 2, 2] }
        model = ResNet(**net_args)
    if args['cuda']:
        model.cuda()
        
    datum = next(iter(train_loader))[0]
    summary(model.cuda(), datum.shape[1:]) # do not feed the batch size

    # sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=args['cuda'])
    sampler = GaussianSampler(mu=0, sigma=1, cuda=args['cuda'])
    optimizer = SimulatedAnnealing(model.parameters(), sampler=sampler)
    if args['sgd']:
        optimizer = optim.Adam(model.parameters(), 
                                lr=1e-3, 
                                betas=(0.9, 0.99), 
                                eps=1e-8, 
                                weight_decay=0) 

    loss_vals = []
    test_loss_vals = 0.
    accuracy = 0.

    for epoch in range(1, args['epochs'] + 1):
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                outputs = model(data)
                loss = F.nll_loss(outputs, target)
                loss_vals.append(loss.detach())

                def closure(weight=None):
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()

                    if (weight is not None):
                        state_dict = model.state_dict()
                        keys = list(state_dict.keys())

                        for key_idx, key in enumerate(keys):
                            state_dict[key] = weight[0]['params'][key_idx]
                            model.load_state_dict(state_dict)
                    outputs = model(data)
                    loss = F.nll_loss(outputs, target)
                    
                    if loss.requires_grad:
                        loss.backward(retain_graph=True)
                    return loss

                optimizer.step(closure)
                if batch_idx % args['log_interval'] == 0:
                    desc = f"Epoch [{epoch}/{args['epochs']}]     "
                    desc += f"[Steps {batch_idx * len(data)} / {len(train_loader.dataset)}]    "
                    desc += f"Batch Loss Train: {loss.item():.4f}     " 
                    pbar.set_description(desc)
                    pbar.update(args['log_interval'])
                elif  batch_idx==len(train_loader)-1:
                    desc = f"Epoch [{epoch}/{args['epochs']}]     "
                    desc += f"[Steps {len(train_loader.dataset)} / {len(train_loader.dataset)}]    "
                    desc += f"Batch Loss Train: {loss.item():.4f}     " 
                    pbar.set_description(desc)
                    pbar.update(args['log_interval'])
    print("\nDone training.")

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss_vals += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        accuracy += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss_vals /= len(test_loader.dataset)    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss_vals, accuracy, len(test_loader.dataset),
            100. * accuracy / len(test_loader.dataset)))
# %%
