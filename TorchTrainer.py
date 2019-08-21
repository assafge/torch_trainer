import torch


class TorchTrainer:
    """wrapper class for torch models training"""
    def __init__(self, cfg:dict, gpu_index: int, logger_obj):
        self.logger = logger_obj
        self.device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
