from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PatchifiedCIFAR10(datasets.CIFAR10):
    def __init__(self, patch_len, root, train=True, transform=None, target_transform=None, download=False):
        super(PatchifiedCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.patch_len = patch_len
    
    def __getitem__(self, index):
        # (seq_len, patch_size_flat)
        # TODO: Implement this function
        ...


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = PatchifiedCIFAR10(patch_size=(16, 16), root='../data', train=True, transform=transform, download=True)
