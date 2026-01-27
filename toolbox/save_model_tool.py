import os
import shutil

import torch


def save_checkpoint(state, is_best, filename="model_last.pth.tar", best_file_name="model_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)