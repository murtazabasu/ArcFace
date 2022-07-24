import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_SIZE = 0.2