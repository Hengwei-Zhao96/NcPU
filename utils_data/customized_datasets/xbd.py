import os
from PIL import Image

from torchvision.datasets import ImageFolder

def xbd_loader(path):
    return Image.open(path)

class xBDDataset(ImageFolder):

    def __init__(self, root, dataset_name) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "xBD", dataset_name)

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        super().__init__(self._base_folder, loader=xbd_loader)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._base_folder)
    