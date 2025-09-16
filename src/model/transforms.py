from torchvision import transforms

#the most general normalization values
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size=224):
    """
     augmentation and normalization for train images
    :param img_size:
    :return:
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,std=NORMALIZE_STD)
    ])

def get_val_transforms(img_size=224):
    """
     for test and val images
    :param img_size:
    :return:
    """

    return transforms.Compose([
        transforms.Resize(int(img_size/0.875)), # 256 is an accepted size -- 224/256 = 0.875
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN,std=NORMALIZE_STD)
    ])

#test block
if __name__ == "__main__":
    print("Train transforms...")
    train_transform = get_train_transforms()
    print(train_transform)

    print("\nVal transforms...")
    val_transform = get_val_transforms()
    print(val_transform)

    print("\n------------- Finished -------------")