import torch
print(torch.cuda.is_available())  # This should return True if CUDA is available
print(torch.cuda.device_count())  # This should return the number of available GPUs


import pytorch_lightning as pl

print(torch.__version__)
print(pl.__version__)