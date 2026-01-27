import os
import tifffile

from torchvision.datasets import ImageFolder

def abcd_loader(path):
    return tifffile.imread(path)

class ABCDDataset(ImageFolder):

    def __init__(self, root) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "ABCD")

        if not self._check_exists():
            raise RuntimeError("ABCD dataset not found.")

        super().__init__(self._base_folder, loader=abcd_loader)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._base_folder)
    