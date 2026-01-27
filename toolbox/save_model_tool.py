import os
import shutil

import torch


def save_checkpoint(state, is_best, filename="model_last.pth.tar", best_file_name="model_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def save_checkpoint_vPU(state, is_lowest_on_val, is_highest_on_test, filepath):
    torch.save(state, os.path.join(filepath, "checkpoint.pth.tar"))
    if is_lowest_on_val:
        shutil.copyfile(os.path.join(filepath, "checkpoint.pth.tar"), os.path.join(filepath, "model_lowest_on_val.pth.tar"))
    if is_highest_on_test:
        shutil.copyfile(os.path.join(filepath, "checkpoint.pth.tar"), os.path.join(filepath, "model_highest_on_test.pth.tar"))
