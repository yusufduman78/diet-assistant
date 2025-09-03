import torchvision

food101dataset = torchvision.datasets.Food101(root="../raw", split="train", download=True)

food_classes = food101dataset.classes

print(food_classes)

print(f"Class number {len(food_classes)}")
