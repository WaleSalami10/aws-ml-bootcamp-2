"""
Generate synthetic cat vs non-cat image dataset for logistic regression.
Creates dummy image data with distinguishable patterns for cats and non-cats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os

def generate_cat_features(num_samples, image_size=(64, 64, 3)):
    """
    Generate synthetic cat images with characteristic features:
    - Triangular ears
    - Circular eyes
    - Whiskers (lines)
    - Warm color palette (oranges, browns)
    """
    height, width, channels = image_size
    images = np.zeros((num_samples, height, width, channels))
    
    for i in range(num_samples):
        # Base warm color background for cats
        base_color = np.random.uniform(0.3, 0.7, 3)  # Warm tones
        base_color[0] += 0.2  # More red/orange
        base_color = np.clip(base_color, 0, 1)
        
        # Create base image
        img = np.random.normal(base_color, 0.1, (height, width, channels))
        img = np.clip(img, 0, 1)
        
        # Add cat-like features
        # Triangular ears (top corners)
        ear_size = np.random.randint(8, 15)
        # Left ear
        for y in range(ear_size):
            for x in range(y, ear_size):
                if y < height and x < width:
                    img[y, x] = np.clip(img[y, x] + 0.3, 0, 1)
        
        # Right ear
        for y in range(ear_size):
            for x in range(ear_size - y, ear_size):
                if y < height and width - x - 1 >= 0:
                    img[y, width - x - 1] = np.clip(img[y, width - x - 1] + 0.3, 0, 1)
        
        # Eyes (circular bright spots)
        eye_y = height // 3
        left_eye_x = width // 3
        right_eye_x = 2 * width // 3
        eye_radius = np.random.randint(3, 7)
        
        for dy in range(-eye_radius, eye_radius + 1):
            for dx in range(-eye_radius, eye_radius + 1):
                if dx*dx + dy*dy <= eye_radius*eye_radius:
                    # Left eye
                    if 0 <= eye_y + dy < height and 0 <= left_eye_x + dx < width:
                        img[eye_y + dy, left_eye_x + dx] = [0.9, 0.9, 0.2]  # Bright yellow
                    # Right eye
                    if 0 <= eye_y + dy < height and 0 <= right_eye_x + dx < width:
                        img[eye_y + dy, right_eye_x + dx] = [0.9, 0.9, 0.2]  # Bright yellow
        
        # Whiskers (horizontal lines)
        whisker_y = height // 2
        whisker_length = width // 4
        for offset in [-5, 0, 5]:
            if 0 <= whisker_y + offset < height:
                # Left whiskers
                img[whisker_y + offset, :whisker_length] = [0.1, 0.1, 0.1]
                # Right whiskers
                img[whisker_y + offset, -whisker_length:] = [0.1, 0.1, 0.1]
        
        images[i] = img
    
    return images

def generate_non_cat_features(num_samples, image_size=(64, 64, 3)):
    """
    Generate synthetic non-cat images with different patterns:
    - Geometric shapes (rectangles, circles)
    - Cool color palette (blues, greens)
    - Random patterns
    """
    height, width, channels = image_size
    images = np.zeros((num_samples, height, width, channels))
    
    for i in range(num_samples):
        # Base cool color background for non-cats
        base_color = np.random.uniform(0.2, 0.6, 3)  # Cool tones
        base_color[2] += 0.2  # More blue
        base_color = np.clip(base_color, 0, 1)
        
        # Create base image
        img = np.random.normal(base_color, 0.1, (height, width, channels))
        img = np.clip(img, 0, 1)
        
        # Add non-cat patterns
        pattern_type = np.random.choice(['rectangle', 'circle', 'stripes'])
        
        if pattern_type == 'rectangle':
            # Add rectangular shapes
            rect_height = np.random.randint(10, 20)
            rect_width = np.random.randint(10, 20)
            start_y = np.random.randint(0, height - rect_height)
            start_x = np.random.randint(0, width - rect_width)
            
            color = np.random.uniform(0, 1, 3)
            img[start_y:start_y+rect_height, start_x:start_x+rect_width] = color
            
        elif pattern_type == 'circle':
            # Add circular shape
            center_y = height // 2
            center_x = width // 2
            radius = np.random.randint(8, 15)
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        if 0 <= center_y + dy < height and 0 <= center_x + dx < width:
                            img[center_y + dy, center_x + dx] = [0.2, 0.8, 0.2]  # Green
        
        else:  # stripes
            # Add horizontal stripes
            stripe_width = 4
            for y in range(0, height, stripe_width * 2):
                if y + stripe_width < height:
                    img[y:y+stripe_width, :] = [0.1, 0.1, 0.8]  # Blue stripes
        
        images[i] = img
    
    return images

def create_dataset(num_cats=500, num_non_cats=500, image_size=(64, 64, 3)):
    """Create complete dataset with cats and non-cats"""
    print(f"Generating {num_cats} cat images...")
    cat_images = generate_cat_features(num_cats, image_size)
    
    print(f"Generating {num_non_cats} non-cat images...")
    non_cat_images = generate_non_cat_features(num_non_cats, image_size)
    
    # Combine datasets
    all_images = np.vstack([cat_images, non_cat_images])
    labels = np.hstack([np.ones(num_cats), np.zeros(num_non_cats)])
    
    # Shuffle the dataset
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    labels = labels[indices]
    
    return all_images, labels

def save_dataset_samples(images, labels, save_dir="StandFord Machine Learning/Supervised Learning/Logistic Regression/images"):
    """Save sample images to visualize the dataset"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure showing sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images from Cat vs Non-Cat Dataset', fontsize=16)
    
    # Show 5 cats and 5 non-cats
    cat_indices = np.where(labels == 1)[0][:5]
    non_cat_indices = np.where(labels == 0)[0][:5]
    
    for i, idx in enumerate(cat_indices):
        axes[0, i].imshow(images[idx])
        axes[0, i].set_title(f'Cat (Label: {int(labels[idx])})')
        axes[0, i].axis('off')
    
    for i, idx in enumerate(non_cat_indices):
        axes[1, i].imshow(images[idx])
        axes[1, i].set_title(f'Non-Cat (Label: {int(labels[idx])})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dataset_samples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample images saved to {save_dir}/dataset_samples.png")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    print("Creating cat vs non-cat dataset...")
    images, labels = create_dataset(num_cats=600, num_non_cats=600, image_size=(64, 64, 3))
    
    # Flatten images for logistic regression (each image becomes a feature vector)
    num_samples, height, width, channels = images.shape
    flattened_images = images.reshape(num_samples, height * width * channels)
    
    # Create train/test split (80/20)
    split_idx = int(0.8 * len(flattened_images))
    
    train_X = flattened_images[:split_idx]
    train_y = labels[:split_idx]
    test_X = flattened_images[split_idx:]
    test_y = labels[split_idx:]
    
    # Save datasets
    data_dir = "StandFord Machine Learning/Data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save training data
    train_df = pd.DataFrame(train_X)
    train_df['label'] = train_y
    train_df.to_csv(f"{data_dir}/cat_classification_train.csv", index=False)
    
    # Save test data
    test_df = pd.DataFrame(test_X)
    test_df['label'] = test_y
    test_df.to_csv(f"{data_dir}/cat_classification_test.csv", index=False)
    
    # Save sample images for visualization
    save_dataset_samples(images, labels)
    
    print(f"\nDataset created successfully!")
    print(f"Training samples: {len(train_X)} (Features: {train_X.shape[1]})")
    print(f"Test samples: {len(test_X)} (Features: {test_X.shape[1]})")
    print(f"Image dimensions: 64x64x3 = {64*64*3} features per image")
    print(f"Class distribution - Cats: {np.sum(labels)}, Non-cats: {len(labels) - np.sum(labels)}")

if __name__ == "__main__":
    main()