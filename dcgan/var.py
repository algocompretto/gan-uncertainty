import torch

NUM_WORKERS = 2
BATCH_SIZE = 16
IMAGE_SIZE = 128
NUM_CHANNELS = 3
NZ = 128
NGF = 64
NDF = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.0002
BETA1 = 0.5
NGPU = 0
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
