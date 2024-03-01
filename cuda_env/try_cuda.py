import torch
import torch.nn as nn

model = nn.Module()
device = torch.device(torch.cuda.current_device())
print(device)

model = model.to(0)
#
# print(torch.cuda.is_available())