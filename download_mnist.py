#!/usr/bin/env python3
"""
Script to download MNIST dataset and convert to .mat format for C++ consumption.
Supports both PyTorch and TensorFlow datasets.
"""

import argparse
import numpy as np
import scipy.io as sio
import os
from pathlib import Path


def download_mnist_torch():
    """Download MNIST using PyTorch."""
    try:
        import torch
        from torchvision import datasets, transforms
        
        print("Downloading MNIST using PyTorch...")
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Download training data
        train_dataset = datasets.MNIST(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Download test data
        test_dataset = datasets.MNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Extract data and labels
        train_data = []
        train_labels = []
        for img, label in train_dataset:
            train_data.append(img.squeeze().numpy())
            train_labels.append(label)
        
        test_data = []
        test_labels = []
        for img, label in test_dataset:
            test_data.append(img.squeeze().numpy())
            test_labels.append(label)
        
        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.int32)
        test_data = np.array(test_data, dtype=np.float32)
        test_labels = np.array(test_labels, dtype=np.int32)
        
        # Normalize to [0, 1] (already done by ToTensor, but ensure it's float32)
        train_data = train_data / 255.0
        test_data = test_data / 255.0
        
        return train_data, train_labels, test_data, test_labels
        
    except ImportError:
        raise ImportError("PyTorch not installed. Install with: pip install torch torchvision")


def download_mnist_tf():
    """Download MNIST using TensorFlow."""
    try:
        import tensorflow as tf
        
        print("Downloading MNIST using TensorFlow...")
        
        # Download MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Convert to float32 and normalize
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        return x_train, y_train, x_test, y_test
        
    except ImportError:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")


def save_to_mat(train_data, train_labels, test_data, test_labels, output_dir="data"):
    """Save MNIST data to .mat files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving data to {output_path}...")
    
    # Save training data
    train_file = output_path / "mnist_train.mat"
    sio.savemat(
        str(train_file),
        {
            'images': train_data,
            'labels': train_labels,
            'num_samples': train_data.shape[0],
            'image_height': train_data.shape[1],
            'image_width': train_data.shape[2]
        }
    )
    print(f"Saved training data: {train_file}")
    print(f"  Shape: {train_data.shape}, Labels: {train_labels.shape}")
    
    # Save test data
    test_file = output_path / "mnist_test.mat"
    sio.savemat(
        str(test_file),
        {
            'images': test_data,
            'labels': test_labels,
            'num_samples': test_data.shape[0],
            'image_height': test_data.shape[1],
            'image_width': test_data.shape[2]
        }
    )
    print(f"Saved test data: {test_file}")
    print(f"  Shape: {test_data.shape}, Labels: {test_labels.shape}")


def main():
    parser = argparse.ArgumentParser(description='Download MNIST and convert to .mat format')
    parser.add_argument(
        '--source',
        choices=['torch', 'tf', 'auto'],
        default='auto',
        help='Source to download from: torch, tf, or auto (default: auto)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for .mat files (default: data)'
    )
    
    args = parser.parse_args()
    
    # Try to download based on source preference
    if args.source == 'torch':
        train_data, train_labels, test_data, test_labels = download_mnist_torch()
    elif args.source == 'tf':
        train_data, train_labels, test_data, test_labels = download_mnist_tf()
    else:  # auto
        try:
            train_data, train_labels, test_data, test_labels = download_mnist_torch()
        except ImportError:
            print("PyTorch not available, trying TensorFlow...")
            train_data, train_labels, test_data, test_labels = download_mnist_tf()
    
    # Save to .mat files
    save_to_mat(train_data, train_labels, test_data, test_labels, args.output_dir)
    
    print("\nDone! MNIST data saved in .mat format.")


if __name__ == "__main__":
    main()
