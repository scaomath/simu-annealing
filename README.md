# Simulated Annealing
An implemention of Simulated Annealing using PyTorch Optimizer interface

## Usage
You need to define a sampler, e.g.:

```python
sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=args.cuda)
# or
sampler = GaussianSampler(mu=0, sigma=1, cuda=args.cuda)
```

The sampler is used for the annealing schedule for Simulated Annealing.
The optimizer needs a `closure()` wrapper with a mechanism to update parameters within the`step` call:
```python
optimizer = SimulatedAnnealing(model.parameters(), sampler=sampler)
def closure(weight=None):
    if weight is not None:
        '''
        update weight here to the state dict
        '''
    output = model(data)
    loss = F.nll_loss(output, target)
    return loss

optimizer.step(closure)
```

## Reference
The code is modified from https://github.com/jramapuram/SimulatedAnnealing