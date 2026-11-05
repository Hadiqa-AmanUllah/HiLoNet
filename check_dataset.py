"""
Check XRF Dataset - Analyze split data sizes and check for overlaps.

This script reads data from the XRF_dataset folder structure and:
1. Reports the number of files in each split (train/val/test/finetune for WiFi, RFID, mmWave)
2. Checks if there are any overlapping files between splits
3. Also checks the raw dataset folder structure
4. Checks for missing files (files in txt but not in dataset folders)
5. Checks for corrupted .npy files

Usage:
    python check_dataset.py                    # Check XRF_dataset (train/val/test/finetune)
    python check_dataset.py --raw              # Check the raw dataset
    python check_dataset.py --both             # Check both datasets
    python check_dataset.py --subjects         # Print unique subject names
    python check_dataset.py --check-missing    # Check for missing files
    python check_dataset.py --check-corrupted  # Check for corrupted files
    python check_dataset.py --check-corrupted --fix  # Check and remove corrupted files
"""

import os
import sys
import numpy as np

from split_train_test import RAW_DATA_ROOT, SPLIT_OUTPUT_ROOT


def extract_subject_name(file_name):
    """Extract subject identifier from filename.

    Filename format: {subject}_{action}_{sequence}  e.g. 28_01_01.npy
    The subject is the first underscore-separated token (index 0).
    """
    base_name = os.path.basename(file_name)
    stem = os.path.splitext(base_name)[0]
    return stem.split('_')[0]


def get_file_set(data_path, subfolder):
    """Get set of file names from a specific subfolder."""
    full_path = os.path.join(data_path, subfolder)
    if not os.path.exists(full_path):
        return set(), 0
    
    files = os.listdir(full_path)
    # Filter out any non-file items (like .DS_Store on Mac)
    file_set = set(f for f in files if os.path.isfile(os.path.join(full_path, f)) and not f.startswith('.'))
    return file_set, len(file_set)


def collect_split_files(data_path, split_name, modality):
    """Collect files for one split+modality from scene-aware or legacy-flat layout.

    Returns:
        keys_to_path: dict where key is canonical sample id and value is absolute npy path
                      scene-aware key format: sceneX/filename.npy
                      flat key format: filename.npy
    """
    split_root = os.path.join(data_path, split_name)
    keys_to_path = {}
    shape_hist = {}

    if not os.path.isdir(split_root):
        return keys_to_path

    scene_dirs = [d for d in sorted(os.listdir(split_root)) if d.lower().startswith('scene')]
    has_scene_layout = any(os.path.isdir(os.path.join(split_root, d, modality)) for d in scene_dirs)
    if has_scene_layout:
        for scene in scene_dirs:
            mod_dir = os.path.join(split_root, scene, modality)
            if not os.path.isdir(mod_dir):
                continue
            for fname in os.listdir(mod_dir):
                fpath = os.path.join(mod_dir, fname)
                if fname.endswith('.npy') and os.path.isfile(fpath):
                    try:
                        arr = np.load(fpath)
                        shape = tuple(arr.shape)
                        shape_hist[shape] = shape_hist.get(shape, 0) + 1
                        key = f'{scene}/{fname}'
                        keys_to_path[key] = fpath
                    except Exception as e:
                        print(f"[Corrupted] {fpath}: {e}")
    return keys_to_path
def get_shape_distribution(file_paths):
    """
    Given a list of .npy file paths, load each and collect shape histogram.
    Returns: dict mapping shape tuple to count
    """
    shape_hist = {}
    for fpath in file_paths:
        try:
            arr = np.load(fpath)
            shape = tuple(arr.shape)
            shape_hist[shape] = shape_hist.get(shape, 0) + 1
        except Exception as e:
            print(f"[Corrupted] {fpath}: {e}")
    return shape_hist


def print_shape_hist(title, shape_hist):
    print(title)
    if not shape_hist:
        print("    - No files")
        return
    for shape, cnt in sorted(shape_hist.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {shape}: {cnt}")


def print_unique_subject_names(data_path=SPLIT_OUTPUT_ROOT):
    """Print unique subject names derived from the first token in each filename."""
    print("=" * 60)
    print("UNIQUE SUBJECT NAMES")
    print("=" * 60)
    print(f"\nDataset path: {os.path.relpath(data_path)}")

    modalities = ["WiFi", "RFID", "mmWave"]
    splits = ["train_data", "val_data", "test_data", "finetune_data"]
    all_subjects = set()

    for split in splits:
        split_subjects = set()
        for modality in modalities:
            keys_to_path = collect_split_files(data_path, split, modality)
            for sample_key in keys_to_path.keys():
                sample_name = os.path.basename(sample_key)
                split_subjects.add(extract_subject_name(sample_name))

        all_subjects.update(split_subjects)
        print(f"\n{split}: {len(split_subjects)} unique subject(s)")
        for subject in sorted(split_subjects):
            print(f"  - {subject}")

    print("\n" + "-" * 60)
    print(f"All splits combined: {len(all_subjects)} unique subject(s)")
    for subject in sorted(all_subjects):
        print(f"  - {subject}")
    print("=" * 60)


def analyze_processed_dataset(data_path=SPLIT_OUTPUT_ROOT):
    """Analyze processed XRF dataset (train/val/test/finetune) and check overlaps + shapes."""
    
    print("=" * 60)
    print("XRF Dataset Analysis (Processed - train/val/test/finetune)")
    print("=" * 60)
    print(f"\nDataset path: {os.path.relpath(data_path)}")
    print()
    
    modalities = ["WiFi", "RFID", "mmWave"]
    splits = ["train_data", "val_data", "test_data", "finetune_data"]
    
    stats = {}
    
    print("-" * 60)
    print("DATA SIZE SUMMARY")
    print("-" * 60)
    
    totals = {s: 0 for s in splits}

    for mod in modalities:
        stats[mod] = {}
        print(f"\n{mod}:")

        mod_total = 0
        for split in splits:
            keys_to_path = collect_split_files(data_path, split, mod)
            file_keys = set(keys_to_path.keys())
            count = len(file_keys)
            shape_hist = get_shape_distribution(list(keys_to_path.values()))

            stats[mod][split] = {
                'count': count,
                'files': file_keys,
                'shape_hist': shape_hist,
            }
            totals[split] += count
            mod_total += count

            print(f"  {split}: {count} files")
            print_shape_hist(f"    Shapes in {split}:", shape_hist)

        print(f"  Total: {mod_total}")
    
    print("\n" + "-" * 60)
    print("OVERALL STATISTICS")
    print("-" * 60)
    print(f"Total train files: {totals['train_data']}")
    print(f"Total val files: {totals['val_data']}")
    print(f"Total test files: {totals['test_data']}")
    print(f"Total finetune files: {totals['finetune_data']}")
    print(f"Total files: {sum(totals.values())}")
    
    print("\n" + "=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)
    
    overlaps_found = False
    
    for mod in modalities:
        train_files = stats[mod]['train_data']['files']
        val_files = stats[mod]['val_data']['files']
        test_files = stats[mod]['test_data']['files']
        finetune_files = stats[mod]['finetune_data']['files']

        overlap_train_val = train_files.intersection(val_files)
        overlap_train_test = train_files.intersection(test_files)
        overlap_train_finetune = train_files.intersection(finetune_files)
        overlap_val_test = val_files.intersection(test_files)
        overlap_val_finetune = val_files.intersection(finetune_files)
        overlap_test_finetune = test_files.intersection(finetune_files)
        overlap_all_four = train_files.intersection(val_files).intersection(test_files).intersection(finetune_files)

        print(f"\n{mod}:")
        print(f"  train ∩ val: {len(overlap_train_val)}")
        print(f"  train ∩ test: {len(overlap_train_test)}")
        print(f"  train ∩ finetune: {len(overlap_train_finetune)}")
        print(f"  val ∩ test: {len(overlap_val_test)}")
        print(f"  val ∩ finetune: {len(overlap_val_finetune)}")
        print(f"  test ∩ finetune: {len(overlap_test_finetune)}")
        print(f"  train ∩ val ∩ test ∩ finetune: {len(overlap_all_four)}")

        if overlap_train_val or overlap_train_test or overlap_train_finetune or overlap_val_test or overlap_val_finetune or overlap_test_finetune:
            overlaps_found = True
            show_samples = sorted(
                list(
                    overlap_train_val
                    | overlap_train_test
                    | overlap_train_finetune
                    | overlap_val_test
                    | overlap_val_finetune
                    | overlap_test_finetune
                )
            )[:20]
            print("  Sample overlapping files (up to 20):")
            for f in show_samples:
                print(f"    - {f}")
    
    print("\n" + "-" * 60)
    print("CROSS-MODALITY CHECK (Same filename across modalities)")
    print("-" * 60)
    
    for split in splits:
        split_sets = {mod: stats[mod][split]['files'] for mod in modalities}
        split_intersection = split_sets["WiFi"]
        for mod in modalities:
            split_intersection = split_intersection.intersection(split_sets[mod])

        if split_intersection:
            print(f"\n{split}: files present in ALL modalities: {len(split_intersection)}")
        else:
            print(f"\n{split}: no files present in all modalities.")

        for mod in modalities:
            missing_vs_wifi = split_sets["WiFi"] - split_sets[mod]
            if mod != "WiFi" and missing_vs_wifi:
                print(f"  {split} -> files in WiFi but missing in {mod}: {len(missing_vs_wifi)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if overlaps_found:
        print("WARNING: Overlapping files found between split sets!")
        print("   This may lead to data leakage and inflated evaluation metrics.")
    else:
        print("OK: No overlapping files found between train/test/finetune sets.")
        print("   Data split appears to be proper.")

    print("Note: shape histograms above show overall tensor shapes per split and modality.")

    print("\n" + "=" * 60)
    print("UNIQUE SUBJECT NAMES")
    print("=" * 60)
    modalities = ["WiFi", "RFID", "mmWave"]
    all_subjects: set = set()
    for split in splits:
        split_subjects: set = set()
        for mod in modalities:
            for sample_key in stats.get(mod, {}).get(split, {}).get('files', set()):
                split_subjects.add(extract_subject_name(os.path.basename(sample_key)))
        all_subjects.update(split_subjects)
        if split_subjects:
            print(f"\n{split} ({len(split_subjects)} subject(s)): {', '.join(sorted(split_subjects))}")
    print(f"\nAll splits combined ({len(all_subjects)} unique subject(s)): {', '.join(sorted(all_subjects))}")
    print("=" * 60)

    return stats


def analyze_raw_dataset(data_path=RAW_DATA_ROOT):
    """Analyze the raw XRF dataset folder structure."""
    
    print("=" * 60)
    print("XRF Raw Dataset Analysis")
    print("=" * 60)
    print(f"\nDataset path: {os.path.relpath(data_path)}")
    print()
    
    if not os.path.exists(data_path):
        print(f"ERROR: Raw dataset path does not exist: {data_path}")
        return None
    
    scenes = [d for d in os.listdir(data_path) 
              if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')]
    
    print(f"Found {len(scenes)} scene(s): {scenes}")
    print()
    
    modalities = ["WiFi", "RFID", "mmWave"]
    all_stats = {}
    
    for scene in scenes:
        print("-" * 60)
        print(f"Scene: {scene}")
        print("-" * 60)
        
        scene_path = os.path.join(data_path, scene)
        
        # Check for nested Scene folder (Scene1/Scene1/)
        nested_scenes = [d for d in os.listdir(scene_path) 
                        if os.path.isdir(os.path.join(scene_path, d)) and not d.startswith('.')]
        
        if nested_scenes:
            # Use first nested scene (usually Scene1/Scene1/)
            nested_path = os.path.join(scene_path, nested_scenes[0])
            print(f"  Using nested path: {nested_scenes[0]}/")
        else:
            nested_path = scene_path
        
        scene_stats = {}
        
        for mod in modalities:
            mod_path = os.path.join(nested_path, mod)
            if os.path.exists(mod_path):
                files = [f for f in os.listdir(mod_path) 
                        if os.path.isfile(os.path.join(mod_path, f)) and not f.startswith('.')]
                scene_stats[mod] = len(files)
                print(f"  {mod}: {len(files)} files")
            else:
                scene_stats[mod] = 0
                print(f"  {mod}: NOT FOUND")
        
        all_stats[scene] = scene_stats
    
    print("\n" + "=" * 60)
    print("RAW DATASET SUMMARY")
    print("=" * 60)
    
    for scene, stats in all_stats.items():
        total = sum(stats.values())
        print(f"{scene}: WiFi={stats['WiFi']}, RFID={stats['RFID']}, mmWave={stats['mmWave']} (Total: {total})")
    
    print("=" * 60)
    
    return all_stats


def check_missing_files(data_path=SPLIT_OUTPUT_ROOT):
    """Check that every sample in the dml_*.txt label files exists and loads, in every modality."""
    print("=" * 60)
    print("CHECKING FOR MISSING FILES")
    print("=" * 60)

    total_missing = 0
    for split, label_name in (("train_data", "dml_train.txt"), ("val_data", "dml_val.txt"), ("test_data", "dml_test.txt")):
        label_path = os.path.join(data_path, label_name)
        if not os.path.exists(label_path):
            print(f"\n{label_name}: not found. Run generate_txt.py first.")
            continue
        with open(label_path) as f:
            names = {line.split(',')[0] + '.npy' for line in f if line.strip()}

        print(f"\n{split} ({len(names)} samples in {label_name}):")
        for modality in ("WiFi", "RFID", "mmWave"):
            missing = sorted(names - set(collect_split_files(data_path, split, modality)))
            total_missing += len(missing)
            print(f"  {modality}: {'OK' if not missing else f'{len(missing)} MISSING'}")
            for name in missing[:10]:
                print(f"    - {name}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

    print("\n" + ("All labeled files are present." if total_missing == 0 else f"Files missing in total: {total_missing}"))
    return total_missing


def check_corrupted_files(data_path=SPLIT_OUTPUT_ROOT, fix=False):
    """
    Check for corrupted .npy files in the dataset.
    
    Attempts to load each .npy file and catches:
    - ValueError: shape mismatch (wrong dimensions)
    - OSError: file not found or unreadable
    - EOFError: truncated file
    - Other loading errors
    
    Args:
        data_path: Path to the XRF_dataset folder
        fix: If True, remove corrupted files. If False, just report them.
    
    Returns:
        Dictionary with corrupted files information
    """
    
    print("=" * 60)
    print("CHECKING FOR CORRUPTED FILES")
    print("=" * 60)
    print(f"\nDataset path: {os.path.relpath(data_path)}")
    print(f"Fix mode: {'ENABLED - will remove corrupted files' if fix else 'DISABLED - will only report'}")
    print()
    
    expected_shapes = {
        "WiFi": (270, 1000),
        "RFID": None,
        "mmWave": None
    }
    
    modalities = ["WiFi", "RFID", "mmWave"]
    
    corrupted_files = {
        "WiFi": {"train": [], "test": []},
        "RFID": {"train": [], "test": []},
        "mmWave": {"train": [], "test": []}
    }
    
    valid_files = {
        "WiFi": {"train": 0, "test": 0},
        "RFID": {"train": 0, "test": 0},
        "mmWave": {"train": 0, "test": 0}
    }
    
    for split in ["train_data", "test_data"]:
        split_name = "train" if split == "train_data" else "test"
        
        print("-" * 60)
        print(f"CHECKING {split.upper()}")
        print("-" * 60)
        
        for mod in modalities:
            mod_path = os.path.join(data_path, split, mod)
            
            if not os.path.exists(mod_path):
                print(f"\n{mod} ({split}): Folder not found - skipping")
                continue
            
            print(f"\n{mod} ({split}):")
            
            npy_files = [f for f in os.listdir(mod_path) if f.endswith('.npy')]
            print(f"  Total files: {len(npy_files)}")
            
            expected_shape = expected_shapes.get(mod)
            mod_corrupted = []
            
            for filename in npy_files:
                filepath = os.path.join(mod_path, filename)
                try:
                    data = np.load(filepath)
                    
                    if expected_shape is not None:
                        expected_size = expected_shape[0] * expected_shape[1]
                        actual_size = data.size
                        
                        if actual_size != expected_size:
                            actual_shape = data.shape if data.ndim > 0 else f"scalar({data})"
                            error_msg = f"Shape mismatch: expected {expected_shape} ({expected_size} elements), got {actual_shape} ({actual_size} elements)"
                            mod_corrupted.append({
                                'file': filename,
                                'error': error_msg,
                                'path': filepath
                            })
                            print(f"    CORRUPTED: {filename}")
                            print(f"      -> {error_msg}")
                        else:
                            valid_files[mod][split_name] += 1
                    else:
                        valid_files[mod][split_name] += 1
                        
                except ValueError as e:
                    error_msg = f"ValueError: {str(e)}"
                    mod_corrupted.append({'file': filename, 'error': error_msg, 'path': filepath})
                    print(f"    CORRUPTED: {filename}")
                    print(f"      -> {error_msg}")
                except OSError as e:
                    error_msg = f"OSError: {str(e)}"
                    mod_corrupted.append({'file': filename, 'error': error_msg, 'path': filepath})
                    print(f"    CORRUPTED: {filename}")
                    print(f"      -> {error_msg}")
                except EOFError as e:
                    error_msg = f"EOFError: {str(e)}"
                    mod_corrupted.append({'file': filename, 'error': error_msg, 'path': filepath})
                    print(f"    CORRUPTED: {filename}")
                    print(f"      -> {error_msg}")
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    mod_corrupted.append({'file': filename, 'error': error_msg, 'path': filepath})
                    print(f"    CORRUPTED: {filename}")
                    print(f"      -> {error_msg}")
            
            corrupted_files[mod][split_name] = mod_corrupted
            
            if mod_corrupted:
                print(f"  -> Found {len(mod_corrupted)} corrupted file(s)")
            else:
                print(f"  -> All {valid_files[mod][split_name]} files OK")
            
            if fix and mod_corrupted:
                print(f"  -> Removing {len(mod_corrupted)} corrupted file(s)...")
                for corrupted in mod_corrupted:
                    try:
                        os.remove(corrupted['path'])
                        print(f"      Removed: {corrupted['file']}")
                    except Exception as e:
                        print(f"      Failed to remove {corrupted['file']}: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_corrupted = 0
    for mod in modalities:
        train_count = len(corrupted_files[mod]["train"])
        test_count = len(corrupted_files[mod]["test"])
        total = train_count + test_count
        total_corrupted += total
        if total > 0:
            print(f"{mod}: Train={train_count}, Test={test_count}, Total={total} corrupted")
    
    if total_corrupted == 0:
        print("No corrupted files found!")
    else:
        print(f"\nTotal corrupted files: {total_corrupted}")
        if fix:
            print("Corrupted files have been removed.")
        else:
            print("Run with --fix to remove corrupted files automatically.")
    
    print("=" * 60)
    
    return corrupted_files


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--raw":
            analyze_raw_dataset()
        elif sys.argv[1] == "--both":
            print("\n" + "#" * 60)
            print("# CHECKING RAW DATASET")
            print("#" * 60 + "\n")
            analyze_raw_dataset()
            print("\n" + "#" * 60)
            print("# CHECKING PROCESSED DATASET (train/test)")
            print("#" * 60 + "\n")
            analyze_processed_dataset()
        elif sys.argv[1] == "--subjects":
            print_unique_subject_names()
        elif sys.argv[1] == "--check-missing":
            check_missing_files()
        elif sys.argv[1] == "--check-corrupted":
            fix_mode = len(sys.argv) > 2 and sys.argv[2] == "--fix"
            check_corrupted_files(fix=fix_mode)
        else:
            print("Usage:")
            print("  python check_dataset.py                       # Check processed dataset")
            print("  python check_dataset.py --raw                 # Check raw dataset")
            print("  python check_dataset.py --both                # Check both datasets")
            print("  python check_dataset.py --subjects            # Print unique subject names")
            print("  python check_dataset.py --check-missing       # Check for missing files")
            print("  python check_dataset.py --check-corrupted     # Check for corrupted files")
            print("  python check_dataset.py --check-corrupted --fix  # Check and remove corrupted files")
    else:
        analyze_processed_dataset()

