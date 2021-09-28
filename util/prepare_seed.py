def prepare_seed(seed, deterministic=True):
    import torch
    import random
    import numpy as np

    if deterministic:
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True