import torch
from torch import distributions as dist

device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_tensor_type(torch.DoubleTensor)

neg_inf = torch.tensor([-torch.inf])
pos_inf = torch.tensor([torch.inf])