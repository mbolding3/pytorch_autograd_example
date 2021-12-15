# pytorch_autograd_example

Repo for learning Pytorch integration.

## Installation

### CPU

```
git clone https://github.com/mbolding3/pytorch_autograd_example.git
cd pytorch_autograd_example
conda create --name learn-pytorch --file conda.txt
```

### GPU

Same as CPU except the last command becomes
```
conda create --name learn-pytorch --file cusignal.txt
```

## Usage

```
conda activate learn-pytorch
cd src
./naive-corr.py
```
or any other code sample you desire.
