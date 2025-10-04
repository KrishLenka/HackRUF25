"""
Script to organize and split dataset for training
Helps organize downloaded images into train/val/test splits
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import argparse

def create_directory_structure(base_dir):
    """Create the expected directory structure for training"""
    base_dir = Path(base_dir)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {base_dir}")
    return base_dir


def organize_binary_dataset(source_healthy, source_unhealthy, target_dir, 
                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize a binary classification dataset (healthy vs unhealthy)
    
    Args:
        source_healthy: Path to directory with healthy images
        source_unhealthy: Path to directory with unhealthy images
        target_dir: Target directory for organized dataset
        train_ratio: Proportion of data for training (default 0.7)
        val_ratio: Proportion of data for validation (default 0.15)
        test_ratio: Proportion of data for testing (default 0.15)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    source_healthy = Path(source_healthy)
    source_unhealthy = Path(source_unhealthy)
    target_dir = Path(target_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (target_dir / split / 'healthy').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'unhealthy').mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name, source_dir in [('healthy', source_healthy), ('unhealthy', source_unhealthy)]:
        print(f"\nProcessing {class_name} images from {source_dir}...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        images = [f for f in source_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(images)} {class_name} images")
        
        if len(images) == 0:
            print(f"Warning: No images found in {source_dir}")
            continue
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy files to respective directories
        for split_name, split_images in [('train', train_images), 
                                         ('val', val_images), 
                                         ('test', test_images)]:
            target_class_dir = target_dir / split_name / class_name
            
            print(f"  Copying {len(split_images)} images to {split_name}/{class_name}...")
            for img_path in tqdm(split_images):
                target_path = target_class_dir / img_path.name
                shutil.copy2(img_path, target_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        healthy_count = len(list((target_dir / split / 'healthy').glob('*')))
        unhealthy_count = len(list((target_dir / split / 'unhealthy').glob('*')))
        total = healthy_count + unhealthy_count
        
        print(f"\n{split.upper()}:")
        print(f"  Healthy: {healthy_count}")
        print(f"  Unhealthy: {unhealthy_count}")
        print(f"  Total: {total}")
    
    print("\n" + "=" * 60)


def organize_multiclass_dataset(source_dirs, target_dir,
                                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize a multi-class classification dataset
    
    Args:
        source_dirs: Dictionary mapping class names to source directories
                    e.g., {'melanoma': 'path/to/melanoma', 'nevus': 'path/to/nevus'}
        target_dir: Target directory for organized dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    target_dir = Path(target_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in source_dirs.keys():
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    summary = {split: {} for split in ['train', 'val', 'test']}
    
    for class_name, source_dir in source_dirs.items():
        source_dir = Path(source_dir)
        print(f"\nProcessing {class_name} images from {source_dir}...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        images = [f for f in source_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(images)} {class_name} images")
        
        if len(images) == 0:
            print(f"Warning: No images found in {source_dir}")
            continue
        
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy files
        for split_name, split_images in [('train', train_images), 
                                         ('val', val_images), 
                                         ('test', test_images)]:
            target_class_dir = target_dir / split_name / class_name
            
            print(f"  Copying {len(split_images)} images to {split_name}/{class_name}...")
            for img_path in tqdm(split_images):
                target_path = target_class_dir / img_path.name
                shutil.copy2(img_path, target_path)
            
            summary[split_name][class_name] = len(split_images)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name, count in summary[split].items():
            print(f"  {class_name}: {count}")
            total += count
        print(f"  Total: {total}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Organize dataset for training')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Dataset organization mode')
    parser.add_argument('--healthy_dir', type=str, help='Path to healthy images directory')
    parser.add_argument('--unhealthy_dir', type=str, help='Path to unhealthy images directory')
    parser.add_argument('--target_dir', type=str, default='data', 
                       help='Target directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proportion of data for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.mode == 'binary':
        if not args.healthy_dir or not args.unhealthy_dir:
            print("\nFor binary mode, you need to specify both --healthy_dir and --unhealthy_dir")
            print("\nExample usage:")
            print("  python prepare_data.py --mode binary \\")
            print("    --healthy_dir /path/to/healthy/images \\")
            print("    --unhealthy_dir /path/to/unhealthy/images \\")
            print("    --target_dir data")
            return
        
        organize_binary_dataset(
            source_healthy=args.healthy_dir,
            source_unhealthy=args.unhealthy_dir,
            target_dir=args.target_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    
    else:
        print("\nFor multiclass mode, please edit the script to specify your class directories.")
        print("Example in the script:")
        print("""
        source_dirs = {
            'melanoma': 'path/to/melanoma',
            'nevus': 'path/to/nevus',
            'bcc': 'path/to/bcc',
            # add more classes...
        }
        
        organize_multiclass_dataset(
            source_dirs=source_dirs,
            target_dir='data'
        )
        """)


if __name__ == "__main__":
    main()
