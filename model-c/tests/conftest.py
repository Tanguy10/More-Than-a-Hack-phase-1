from possession_c.resources import restrict_resources
restrict_resources()
import torch
torch.set_num_threads(2)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
