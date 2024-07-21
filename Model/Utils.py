from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

GPU = "cuda"
CPU = "cpu"

def cosine_sim(vec1, vec2, is_1d=False):
    if is_1d:
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def get_running_device():
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device(GPU)

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # Else use cpu
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device(CPU)

    return device
