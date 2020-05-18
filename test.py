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

print(out)
