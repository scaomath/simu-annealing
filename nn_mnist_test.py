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

# import Simulated annealing optimizer
from sa import UniformSampler, GaussianSampler
#%%
# Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()

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

torch.manual_seed(args['seed'])
if args['cuda']:
    torch.cuda.manual_seed(args['seed'])

#%%
kwargs = {'num_workers': 1, 'pin_memory': True} if args['cuda'] else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
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
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args['cuda']:
    model.cuda()

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
            print("old = ", loss.item(), "| pert = ", loss_perturbed.item(),
                  " | ap = ", ap.item(), " | tau = ", tau)
            if ap.item() > np.random.rand():
                return loss_perturbed, True

            # return the original loss if above fails
            # or if the temp is now annealed
            return loss, False


sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=args['cuda'])
#sampler = GaussianSampler(mu=0, sigma=1, cuda=args['cuda'])
optimizer = SimulatedAnnealing(model.parameters(), sampler=sampler)
#%%
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        def closure(weight=None):
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            if (weight is not None):
                state_dict = model.state_dict()
                keys = list(state_dict.keys())

                for key_idx, key in enumerate(keys):
                    # print(state_dict[key])
                    # print(weight_new[0]['params'])
                    state_dict[key] = weight[0]['params'][key_idx]
                    model.load_state_dict(state_dict)
            outputs = model(data)
            loss = F.nll_loss(outputs, target)
            
            if loss.requires_grad:
                loss.backward(retain_graph=True)
            return loss

        loss = optimizer.step(closure)
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args['epochs'] + 1):
    train(epoch)
    test()

# %%
