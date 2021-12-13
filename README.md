# pytorch_autograd_example

Repo for learning Pytorch integration.

## Installation

### CPU

```
git clone https://github.com/mbolding3/pytorch_autograd_example.git
cd pytorch_autograd_example
conda create --name pytorch --file conda.txt
```

In order to use the Cusignal examples you will alternatively need `conda create --name pytorch+cusignal --file cusignal.txt`.

### GPU

Same as CPU except the last command becomes
```
conda create --name cusignal+pytorch --file cusignal.txt
```

## Usage

```
conda activate pytorch
```
or
```
conda activate cusignal+pytorch
```
as appropriate, then
```
cd src
./naive-corr.py
```
or any other code sample you desire.
