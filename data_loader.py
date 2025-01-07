import os
import random
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage, Compose, Resize, RandomHorizontalFlip, RandomRotation
from torch.utils.data import DataLoader, random_split
from concurrent.futures import ProcessPoolExecutor


class CropDataLoader:
    def __init__(self, root_dir, transform, batch_size=16, workers=4, pre_fetch=2, train_split=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.workers = workers
        self.pre_fetch = pre_fetch
        self.train_split = train_split

        # Precompute the augmentation transforms for GPU (if needed)
        self.augment_transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(90)
        ])

    def _clean_and_augment(self):
        # Load the dataset
        dataset = ImageFolder(root=self.root_dir)

        # Get the number of images per class
        class_counts = {class_name: 0 for class_name in dataset.classes}
        for _, class_idx in dataset.samples:
            class_counts[dataset.classes[class_idx]] += 1

        # Calculate the average number of images per class
        total_images = sum(class_counts.values())
        num_classes = len(class_counts)
        avg_images = total_images // num_classes  # Average number of images per class

        print(f"Average number of images per class: {avg_images}")

        # Iterate over classes and handle augmentation and cleaning
        for class_name in class_counts:
            class_dir = os.path.join(dataset.root, class_name)
            images = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
            num_images = class_counts[class_name]

            # Delete non-image files
            self._delete_non_images(images)

            # Re-list images after deletion
            images = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir) if
                      fname.lower().endswith(('png', 'jpg', 'jpeg'))]
            num_images = len(images)

            # If the class has more images than the average, delete extra images
            if num_images > avg_images:
                num_to_delete = num_images - avg_images
                images_to_delete = random.sample(images, num_to_delete)
                for img_path in images_to_delete:
                    self._delete_image(img_path)

            # If the class has fewer images than the average, augment it
            elif num_images < avg_images:
                num_to_augment = avg_images - num_images
                for _ in range(num_to_augment):
                    img_path = random.choice(images)
                    self._augment_and_save(img_path, class_dir)

    def _delete_non_images(self, images):
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify the image file
            except (UnidentifiedImageError, IOError):
                # Delete non-image file
                os.remove(img_path)

    def _delete_image(self, img_path):
        os.remove(img_path)

    def _augment_and_save(self, img_path, class_dir):
        image = Image.open(img_path).convert("RGB")

        # List to store augmented images
        augmented_images = []

        # Randomly choose a transformation
        transform_choice = random.choice(['flip_h', 'flip_v', 'rotate'])

        if transform_choice == 'flip_h':
            # Apply horizontal flip
            augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(augmented_image)

        elif transform_choice == 'flip_v':
            # Apply vertical flip
            augmented_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            augmented_images.append(augmented_image)

        elif transform_choice == 'rotate':
            # Apply random rotation (90, 180, or 270 degrees)
            rotation_degree = random.choice([90, 180, 270])
            augmented_image = image.rotate(rotation_degree)
            augmented_images.append(augmented_image)

        # Save the augmented images
        for aug_img in augmented_images:
            augmented_image_pil = ToPILImage()(self.transform(aug_img))  # Apply the transform
            augmented_image_pil.save(os.path.join(class_dir, f"augmented_{random.randint(1000, 9999)}.png"))

    def prepare_data(self):
        # Clean and augment data
        self._clean_and_augment()

        # Reload the dataset to include augmented images
        dataset = ImageFolder(root=self.root_dir, transform=self.transform)

        # Split dataset
        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True, prefetch_factor=self.pre_fetch)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=True, prefetch_factor=self.pre_fetch)

        return train_loader, val_loader, dataset.classes
