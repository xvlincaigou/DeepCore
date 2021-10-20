from torchvision import datasets, transforms

def QMNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]# UNKNOWN
    std = [0.3081]# UNKNOWN
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.QMNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
    dst_test = datasets.QMNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test