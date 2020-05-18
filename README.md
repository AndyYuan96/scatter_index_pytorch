# scatter_index_pytorch



## why to use

​	Pytorch's scatter needs input and output have same dimension.

​	This module support use a 3D tensor to index a 4D tensor in C(feature) channel.

​	NOTE !!!
​		only support CUDA Tensor
​		only support index over feature(C) dim  (B,C,H,W)
​		didn't check whether the index is valid


## how to install

```
env:
	pytorch-1.1

python setup.py install
```



## how to use

```
import torch
from scatter_index import scatter_index
batch = 2
cls = 12
h = 4
w = 4


cls_predict = torch.rand((batch,cls,h,w))
reg_predict = torch.rand((batch,cls,h,w))

_,index = cls_predict.max(dim=1)

out = torch.zeros(index.shape)

reg_predict = reg_predict.cuda()
index = index.cuda()
out = out.cuda()

scatter_index(reg_predict, index, out)
```

