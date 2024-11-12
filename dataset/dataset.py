from torchvision import datasets, transforms

# ViT models typically require images of size 224x224
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for pre-trained models
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = train_dataset[0]
    x = x.permute(1, 2, 0).numpy()
    plt.imshow(x)
    plt.show()
    print(x, y)