"""
Script to organize the multi-class skin condition dataset
- Fixes nested train/train and test/test structure
- Splits Healthy Skin and vitiligo folders into train/test
- Creates proper validation split
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def organize_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize the dataset with 25 classes
    """
    random.seed(seed)
    data_dir = Path(data_dir)
    
    print("=" * 70)
    print("ORGANIZING SKIN CONDITION DATASET (25 CLASSES)")
    print("=" * 70)
    
    # Step 1: Flatten the nested structure (train/train -> train, test/test -> test)
    print("\n1. Fixing nested directory structure...")
    
    train_nested = data_dir / "train" / "train"
    test_nested = data_dir / "test" / "test"
    
    if train_nested.exists():
        print("   Found nested train/train/ - flattening...")
        # Get all condition folders
        condition_folders = [f for f in train_nested.iterdir() if f.is_dir()]
        
        # Move each folder up one level
        for folder in condition_folders:
            target = data_dir / "train" / folder.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(folder), str(target))
            print(f"   ✓ Moved {folder.name}")
        
        # Remove the empty nested "train" folder
        if train_nested.exists() and not list(train_nested.iterdir()):
            train_nested.rmdir()
            print("   ✓ Removed empty nested train folder")
    
    if test_nested.exists():
        print("   Found nested test/test/ - flattening...")
        condition_folders = [f for f in test_nested.iterdir() if f.is_dir()]
        
        for folder in condition_folders:
            target = data_dir / "test" / folder.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(folder), str(target))
            print(f"   ✓ Moved {folder.name}")
        
        if test_nested.exists() and not list(test_nested.iterdir()):
            test_nested.rmdir()
            print("   ✓ Removed empty nested test folder")
    
    # Step 2: Split Healthy Skin folder
    print("\n2. Splitting 'Healthy Skin' folder...")
    healthy_source = data_dir / "Healthy Skin"
    
    if healthy_source.exists():
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        images = [f for f in healthy_source.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"   Found {len(images)} healthy skin images")
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate splits
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create target directories
        train_dir = data_dir / "train" / "Healthy_Skin"
        val_dir = data_dir / "val" / "Healthy_Skin"
        test_dir = data_dir / "test" / "Healthy_Skin"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        print(f"   Copying {len(train_images)} to train...")
        for img in tqdm(train_images, desc="   Train"):
            shutil.copy2(img, train_dir / img.name)
        
        print(f"   Copying {len(val_images)} to val...")
        for img in tqdm(val_images, desc="   Val"):
            shutil.copy2(img, val_dir / img.name)
        
        print(f"   Copying {len(test_images)} to test...")
        for img in tqdm(test_images, desc="   Test"):
            shutil.copy2(img, test_dir / img.name)
        
        print(f"   ✓ Healthy Skin: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Step 3: Split vitiligo folder
    print("\n3. Splitting 'vitiligo' folder...")
    vitiligo_source = data_dir / "vitiligo"
    
    if vitiligo_source.exists():
        images = [f for f in vitiligo_source.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"   Found {len(images)} vitiligo images")
        
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create target directories
        train_dir = data_dir / "train" / "vitiligo"
        val_dir = data_dir / "val" / "vitiligo"
        test_dir = data_dir / "test" / "vitiligo"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        print(f"   Copying {len(train_images)} to train...")
        for img in tqdm(train_images, desc="   Train"):
            shutil.copy2(img, train_dir / img.name)
        
        print(f"   Copying {len(val_images)} to val...")
        for img in tqdm(val_images, desc="   Val"):
            shutil.copy2(img, val_dir / img.name)
        
        print(f"   Copying {len(test_images)} to test...")
        for img in tqdm(test_images, desc="   Test"):
            shutil.copy2(img, test_dir / img.name)
        
        print(f"   ✓ vitiligo: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Step 4: Create validation split for existing 23 conditions
    print("\n4. Creating validation split for existing conditions...")
    
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = data_dir / "val"
    
    # Get all condition folders from train
    condition_folders = [f for f in train_dir.iterdir() 
                        if f.is_dir() and f.name not in ["Healthy_Skin", "vitiligo"]]
    
    print(f"   Found {len(condition_folders)} existing condition folders")
    
    for condition in tqdm(condition_folders, desc="   Creating val splits"):
        source = train_dir / condition.name
        val_target = val_dir / condition.name
        val_target.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this condition
        images = [f for f in source.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if len(images) == 0:
            continue
        
        # Take ~15% for validation
        random.shuffle(images)
        n_val = max(1, int(len(images) * 0.15))  # At least 1 image
        val_images = images[:n_val]
        
        # Move to val
        for img in val_images:
            target = val_target / img.name
            shutil.move(str(img), str(target))
    
    # Step 5: Generate summary
    print("\n" + "=" * 70)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 70)
    
    # Count all classes
    train_classes = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])
    val_classes = sorted([f.name for f in val_dir.iterdir() if f.is_dir()])
    test_classes = sorted([f.name for f in test_dir.iterdir() if f.is_dir()])
    
    print(f"\nTotal classes: {len(train_classes)}")
    print("\nClass list:")
    for i, cls in enumerate(train_classes, 1):
        print(f"  {i:2d}. {cls}")
    
    # Count images per split
    print("\n" + "-" * 70)
    print("DATASET STATISTICS")
    print("-" * 70)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    print(f"\n{'Class':<60} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 70)
    
    for cls in train_classes:
        train_count = len([f for f in (train_dir / cls).glob('*') if f.is_file()])
        val_count = len([f for f in (val_dir / cls).glob('*') if f.is_file()]) if (val_dir / cls).exists() else 0
        test_count = len([f for f in (test_dir / cls).glob('*') if f.is_file()]) if (test_dir / cls).exists() else 0
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        # Truncate long names
        display_name = cls[:57] + "..." if len(cls) > 60 else cls
        print(f"{display_name:<60} {train_count:>8} {val_count:>8} {test_count:>8}")
    
    print("-" * 70)
    print(f"{'TOTAL':<60} {total_train:>8} {total_val:>8} {total_test:>8}")
    print("-" * 70)
    
    # Save class names to file
    classes_file = data_dir.parent / "class_names.txt"
    with open(classes_file, 'w') as f:
        for cls in train_classes:
            f.write(f"{cls}\n")
    
    print(f"\n✓ Class names saved to: {classes_file}")
    print(f"✓ Dataset ready for training with {len(train_classes)} classes!")
    print("\nNext steps:")
    print("  cd backend")
    print(f"  python train.py --num_classes {len(train_classes)} --epochs 50")
    print("=" * 70)


if __name__ == "__main__":
    # Run organization
    organize_dataset(
        data_dir="/Users/klenka/HackRUF25/backend/data",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
