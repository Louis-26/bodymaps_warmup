#!/usr/bin/env python
"""
Postprocessing script for vertebrae segmentation masks.
Applies morphological operations and connected component analysis to reduce errors.
Usage: python postprocessing_vertebrae.py --input_dir /path/to/predictions --output_dir /path/to/output
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
import cc3d
import glob


def clean_small_components(mask, min_size_ratio=0.05):
    """
    Remove small connected components that are likely noise.
    
    Args:
        mask: 3D binary mask (numpy array)
        min_size_ratio: Minimum component size as ratio of largest component
    
    Returns:
        Cleaned binary mask
    """
    # Label connected components
    labeled_array, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Get component sizes
    component_sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
    
    # Find largest component size
    max_size = np.max(component_sizes)
    min_size_threshold = max_size * min_size_ratio
    
    # Keep only components larger than threshold
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        if component_sizes[i] >= min_size_threshold:
            cleaned_mask[labeled_array == i] = 1
    
    return cleaned_mask.astype(np.uint8)


def morphological_closing(mask, kernel_size=5):
    """
    Apply morphological closing to fill small holes.
    
    Args:
        mask: 3D binary mask (numpy array)
        kernel_size: Size of the structuring element
    
    Returns:
        Processed binary mask
    """
    struct = ndimage.generate_binary_structure(3, 1)
    # Dilate then erode (closing)
    closed = ndimage.binary_closing(mask, structure=struct, iterations=2)
    return closed.astype(np.uint8)


def morphological_opening(mask, kernel_size=3):
    """
    Apply morphological opening to remove small protrusions.
    
    Args:
        mask: 3D binary mask (numpy array)
        kernel_size: Size of the structuring element
    
    Returns:
        Processed binary mask
    """
    struct = ndimage.generate_binary_structure(3, 1)
    # Erode then dilate (opening)
    opened = ndimage.binary_opening(mask, structure=struct, iterations=1)
    return opened.astype(np.uint8)


def keep_largest_component(mask):
    """
    Keep only the largest connected component.
    Useful for vertebrae which should be single objects.
    
    Args:
        mask: 3D binary mask (numpy array)
    
    Returns:
        Binary mask with only largest component
    """
    labeled_array, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Find largest component
    component_sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
    largest_component_label = np.argmax(component_sizes)
    
    # Keep only largest component
    largest_mask = (labeled_array == largest_component_label).astype(np.uint8)
    return largest_mask


def postprocess_vertebra(mask_array, strategy='aggressive'):
    """
    Apply postprocessing pipeline to vertebra mask.
    
    Args:
        mask_array: 3D binary mask (numpy array)
        strategy: 'conservative' (minimal processing) or 'aggressive' (strong cleanup)
    
    Returns:
        Postprocessed mask
    """
    if np.sum(mask_array) == 0:
        # Empty mask, return as is
        return mask_array.astype(np.uint8)
    
    # Ensure mask is binary and minimal memory
    processed = (mask_array > 0.5).astype(np.uint8)
    
    if strategy == 'aggressive':
        # Keep only largest connected component first (memory efficient)
        processed = keep_largest_component(processed)
        
        # Fill holes with closing (lite version)
        struct = ndimage.generate_binary_structure(3, 1)
        processed = ndimage.binary_closing(processed, structure=struct, iterations=1).astype(np.uint8)
        
        # Single pass opening to remove small protrusions
        processed = ndimage.binary_opening(processed, structure=struct, iterations=1).astype(np.uint8)
        
    elif strategy == 'conservative':
        # Only fill small holes
        struct = ndimage.generate_binary_structure(3, 1)
        processed = ndimage.binary_closing(processed, structure=struct, iterations=1).astype(np.uint8)
    
    return processed.astype(np.uint8)


def postprocess_multiclass_labels(labels_array, strategy='aggressive', exclude_labels=[0]):
    """
    Apply postprocessing to multi-class label volume (like combined_labels.nii.gz).
    Processes each class separately while preserving labels.
    
    Args:
        labels_array: 3D label array with multiple classes
        strategy: 'conservative' or 'aggressive'
        exclude_labels: Class labels to skip (e.g., [0] for background)
    
    Returns:
        Postprocessed multi-class label array
    """
    processed = labels_array.copy().astype(np.uint8)
    unique_labels = np.unique(labels_array)
    
    struct = ndimage.generate_binary_structure(3, 1)
    
    for label_id in unique_labels:
        if label_id in exclude_labels:
            continue
        
        # Extract binary mask for this class
        class_mask = (labels_array == label_id).astype(np.uint8)
        
        if np.sum(class_mask) == 0:
            continue
        
        if strategy == 'aggressive':
            # Keep only largest component
            class_mask = keep_largest_component(class_mask)
            
            # Fill holes
            class_mask = ndimage.binary_closing(class_mask, structure=struct, iterations=1).astype(np.uint8)
            
            # Remove protrusions
            class_mask = ndimage.binary_opening(class_mask, structure=struct, iterations=1).astype(np.uint8)
            
        elif strategy == 'conservative':
            # Just fill holes
            class_mask = ndimage.binary_closing(class_mask, structure=struct, iterations=1).astype(np.uint8)
        
        # Update processed volume with cleaned class
        # First clear only the voxels that belonged to this class
        processed[labels_array == label_id] = 0
        # Then set the cleaned voxels
        processed[class_mask == 1] = label_id
    
    return processed.astype(np.uint8)


def process_single_file(input_path, output_path, strategy='aggressive'):
    """
    Process a single segmentation file.
    
    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save output .nii.gz file
        strategy: Postprocessing strategy
    
    Returns:
        Boolean indicating success
    """
    try:
        # Load the NIfTI file
        img = nib.load(input_path)
        mask = img.get_fdata()
        affine = img.affine
        
        # Convert to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply postprocessing
        processed_mask = postprocess_vertebra(binary_mask, strategy=strategy)
        
        # Save the result
        output_img = nib.Nifti1Image(processed_mask, affine)
        nib.save(output_img, output_path)
        
        input_voxels = np.sum(binary_mask)
        output_voxels = np.sum(processed_mask)
        change_pct = 100.0 * (input_voxels - output_voxels) / (input_voxels + 1e-8)
        
        print(f"  ✓ {os.path.basename(input_path)}: {input_voxels} → {output_voxels} voxels ({change_pct:.1f}% change)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")
        return False


def process_combined_labels(input_path, output_path, strategy='aggressive'):
    """
    Process combined multi-class label file.
    
    Args:
        input_path: Path to combined_labels.nii.gz
        output_path: Path to save postprocessed combined_labels
        strategy: Postprocessing strategy
    
    Returns:
        Boolean indicating success
    """
    try:
        # Load the NIfTI file
        img = nib.load(input_path)
        labels = img.get_fdata().astype(np.uint8)
        affine = img.affine
        
        # Apply multi-class postprocessing
        processed_labels = postprocess_multiclass_labels(labels, strategy=strategy, exclude_labels=[0])
        
        # Save the result
        output_img = nib.Nifti1Image(processed_labels, affine)
        nib.save(output_img, output_path)
        
        input_voxels = np.sum(labels > 0)
        output_voxels = np.sum(processed_labels > 0)
        change_pct = 100.0 * (input_voxels - output_voxels) / (input_voxels + 1e-8)
        
        print(f"  ✓ {os.path.basename(input_path)}: {input_voxels} → {output_voxels} voxels ({change_pct:.1f}% change)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")
        return False


def process_directory(input_dir, output_dir, pattern='vertebrae_*.nii.gz', strategy='aggressive', process_combined=True):
    """
    Process all vertebrae files in a directory structure.
    
    Args:
        input_dir: Root input directory containing case subdirectories
        output_dir: Root output directory
        pattern: Glob pattern for vertebrae files
        strategy: Postprocessing strategy
        process_combined: Whether to also postprocess combined_labels.nii.gz
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all case directories
    case_dirs = [d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]
    
    if not case_dirs:
        print(f"No case directories found in {input_dir}")
        return
    
    total_files = 0
    processed_files = 0
    
    for case_dir in sorted(case_dirs):
        case_name = os.path.basename(case_dir)
        seg_dir = os.path.join(case_dir, 'segmentations')
        
        print(f"\n📁 Processing case: {case_name}")
        
        # Process individual vertebrae files
        if os.path.isdir(seg_dir):
            vertebrae_files = glob.glob(os.path.join(seg_dir, pattern))
            
            if vertebrae_files:
                output_case_dir = os.path.join(output_dir, case_name, 'segmentations')
                os.makedirs(output_case_dir, exist_ok=True)
                
                print("  Individual vertebrae:")
                for vertebra_file in sorted(vertebrae_files):
                    vertebra_name = os.path.basename(vertebra_file)
                    output_path = os.path.join(output_case_dir, vertebra_name)
                    
                    success = process_single_file(vertebra_file, output_path, strategy=strategy)
                    total_files += 1
                    if success:
                        processed_files += 1
        
        # Process combined_labels if requested
        if process_combined:
            combined_labels_path = os.path.join(case_dir, 'combined_labels.nii.gz')
            if os.path.isfile(combined_labels_path):
                output_case_dir = os.path.join(output_dir, case_name)
                os.makedirs(output_case_dir, exist_ok=True)
                output_path = os.path.join(output_case_dir, 'combined_labels.nii.gz')
                
                print("  Multi-class labels:")
                success = process_combined_labels(combined_labels_path, output_path, strategy=strategy)
                total_files += 1
                if success:
                    processed_files += 1
    
    print(f"\n✅ Postprocessing complete: {processed_files}/{total_files} files processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Postprocess vertebrae segmentation masks and combined labels to reduce errors.'
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Root directory containing case subdirectories with segmentations'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory to save postprocessed masks'
    )
    parser.add_argument(
        '--strategy',
        choices=['conservative', 'aggressive'],
        default='aggressive',
        help='Postprocessing strategy: conservative (minimal) or aggressive (strong cleanup)'
    )
    parser.add_argument(
        '--pattern',
        default='vertebrae_*.nii.gz',
        help='Glob pattern for vertebrae files'
    )
    parser.add_argument(
        '--skip-combined',
        action='store_true',
        help='Skip postprocessing of combined_labels.nii.gz files'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Vertebrae Segmentation Postprocessing")
    print("=" * 70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Strategy:         {args.strategy}")
    print(f"Pattern:          {args.pattern}")
    print(f"Process combined: {not args.skip_combined}")
    print("=" * 70)
    
    process_directory(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        strategy=args.strategy,
        process_combined=not args.skip_combined
    )


if __name__ == '__main__':
    main()
