#!/bin/bash

# Directory containing the files
DATA_DIR="/home/mohammad/projects/HMS/super-resolution/MKH-sr/simulation-motion/repo/data/datasets/volumes/image/"
OUTPUT_DIR="./splits"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all NIfTI (.nii and .nii.gz) files and save basenames to a temporary file
find "$DATA_DIR" -type f \( -name "*.nii" -o -name "*.nii.gz" \) -exec basename {} \; > all_files.txt

# Count total number of files
total_files=$(wc -l < all_files.txt)
echo "Total NIfTI files found: $total_files"

# Calculate fold size (20% of total)
fold_size=$(( (total_files + 4) / 5 ))  # Round up division
echo "Files per fold: $fold_size"

# Create 5 folds
for fold in {1..5}; do
    echo "Creating fold $fold..."
    
    # Create new files for this fold
    train_file="$OUTPUT_DIR/fold${fold}_train.txt"
    test_file="$OUTPUT_DIR/fold${fold}_test.txt"
    
    # Calculate test range for this fold
    start_line=$(( (fold - 1) * fold_size + 1 ))
    end_line=$(( fold * fold_size ))
    
    # Ensure we don't exceed total files in last fold
    if [ $end_line -gt $total_files ]; then
        end_line=$total_files
    fi
    
    # Extract test set (20%)
    sed -n "${start_line},${end_line}p" all_files.txt > "$test_file"
    
    # Extract train set (remaining 80%)
    if [ $fold -eq 1 ]; then
        sed -n "$((end_line + 1)),${total_files}p" all_files.txt > "$train_file"
    elif [ $fold -eq 5 ]; then
        sed -n "1,$((start_line - 1))p" all_files.txt > "$train_file"
    else
        sed -n "1,$((start_line - 1))p" all_files.txt > "$train_file"
        sed -n "$((end_line + 1)),${total_files}p" all_files.txt >> "$train_file"
    fi
    
    # Count files in each set
    train_count=$(wc -l < "$train_file")
    test_count=$(wc -l < "$test_file")
    echo "Fold $fold - Train: $train_count, Test: $test_count files"
done

# Optional: Create YAML configuration file
cat > "$OUTPUT_DIR/data_split.yaml" << EOL
# 5-fold cross-validation splits
fold1:
  train: fold1_train.txt
  test: fold1_test.txt
fold2:
  train: fold2_train.txt
  test: fold2_test.txt
fold3:
  train: fold3_train.txt
  test: fold3_test.txt
fold4:
  train: fold4_train.txt
  test: fold4_test.txt
fold5:
  train: fold5_train.txt
  test: fold5_test.txt
EOL

# Cleanup
rm all_files.txt

echo "Done! Split files are in $OUTPUT_DIR"
