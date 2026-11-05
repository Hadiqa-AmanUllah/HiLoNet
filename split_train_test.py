
import argparse
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

from generate_txt import generate_txt_mix


# The only place data paths are set; every other script imports these two.
RAW_DATA_ROOT = "../Datasets/XRF55"  # Input: raw dataset folder holding Scene1..Scene4
SPLIT_OUTPUT_ROOT = "./dataset/XRF_dataset"  # Output: split folders (existing splits here are replaced)

MODALITIES = ["RFID", "WiFi", "mmWave"]
SCENES = [1, 2, 3, 4]
SPLITS = ["train_data", "val_data", "test_data", "finetune_data"]
SPLIT_NAME_FILE = "split_name.txt"  # Tells training and evaluation which split is active
TRAIN_SAMPLES_PER_ACTIVITY = 14  # Mode 0: later samples of each activity go to validation


def _clear_existing_output(dst_root):
    """Remove existing split output directories to avoid mixing old/new data."""
    for split in SPLITS:
        split_path = os.path.join(dst_root, split)
        if os.path.isdir(split_path):
            shutil.rmtree(split_path)


def _ensure_dirs(dst_root):
    """Create train/test/finetune folder tree with scene and modality subfolders."""
    for split in SPLITS:
        for scene in SCENES:
            for modality in MODALITIES:
                path = os.path.join(dst_root, split, f"scene{scene}", modality)
                os.makedirs(path, exist_ok=True)


def _clear_scene_dirs(dst_root, scene_id):
    """Wipe and recreate all modality sub-dirs for a given scene in all splits."""
    for split in SPLITS:
        for modality in MODALITIES:
            path = os.path.join(dst_root, split, f"scene{scene_id}", modality)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)


def _parse_filename(stem):
    """Parse filename format: subject_activity_sample (without extension)."""
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        subject = int(parts[0])
        activity = int(parts[1])
        sample_idx = int(parts[2])
    except ValueError:
        return None
    return subject, activity, sample_idx


def _list_scene_samples(scene_root):
    """Read sample stems from RFID folder for a scene."""
    rfid_dir = os.path.join(scene_root, "RFID")
    if not os.path.isdir(rfid_dir):
        return []
    stems = [file[:-4] for file in os.listdir(rfid_dir) if file.endswith(".npy") and _parse_filename(file[:-4])]
    return sorted(stems)


def _link_modalities(scene_root, dst_root, split_name, scene_id, stem):
    """Symlink one sample across all modalities into its split folder."""
    for modality in MODALITIES:
        src = os.path.join(scene_root, modality, f"{stem}.npy")
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_root, split_name, f"scene{scene_id}", modality, f"{stem}.npy")
        if os.path.exists(dst):
            continue
        os.symlink(os.path.abspath(src), dst)


def _count_npy_files(dir_path):
    """Count .npy files directly inside a directory."""
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for name in os.listdir(dir_path) if name.endswith('.npy'))


def _collect_unique_subjects(dir_path):
    """Collect unique subject ids from .npy filenames directly inside a directory."""
    if not os.path.isdir(dir_path):
        return []
    subjects = set()
    for name in os.listdir(dir_path):
        if not name.endswith('.npy'):
            continue
        parsed = _parse_filename(name[:-4])
        if parsed is None:
            continue
        subject, *_ = parsed
        subjects.add(subject)
    return sorted(subjects)


def _print_directory_sample_counts(dst_root):
    """Print per-directory sample counts for all splits/scenes/modalities."""
    print("\nDirectory sample counts:")
    for split in SPLITS:
        print(f"\n[{split}]")
        split_total = 0
        for scene_id in SCENES:
            scene_total = 0
            print(f"  scene{scene_id}:")
            for modality in MODALITIES:
                dir_path = os.path.join(dst_root, split, f"scene{scene_id}", modality)
                count = _count_npy_files(dir_path)
                scene_total += count
                print(f"    {modality}: {count}")
            split_total += scene_total
            print(f"    Total: {scene_total}")
        print(f"  {split} total: {split_total}")


def _print_unique_subjects_by_directory(dst_root):
    """Print unique subjects found in each split/scene directory."""
    print("\nUnique subjects by directory:")
    for split in SPLITS:
        print(f"\n[{split}]")
        split_subjects = set()
        for scene_id in SCENES:
            scene_subjects = set()
            for modality in MODALITIES:
                dir_path = os.path.join(dst_root, split, f"scene{scene_id}", modality)
                scene_subjects.update(_collect_unique_subjects(dir_path))

            split_subjects.update(scene_subjects)
            if scene_subjects:
                scene_subjects_sorted = sorted(scene_subjects)
                print(
                    f"  scene{scene_id}: {len(scene_subjects_sorted)} subject(s) -> "
                    f"{scene_subjects_sorted}"
                )
            else:
                print(f"  scene{scene_id}: 0 subject(s) -> []")

        split_subjects_sorted = sorted(split_subjects)
        print(
            f"  {split} unique subjects ({len(split_subjects_sorted)}): "
            f"{split_subjects_sorted}"
        )


def split_by_subject_across_scenes(
    raw_root=RAW_DATA_ROOT,
    dst_root=SPLIT_OUTPUT_ROOT,
    mode=0,
):
    """
    Create train/val/test split for Scene1..Scene4, then write the label files.

    Scene 1, with subjects in order of subject id (the paper's splits):
        Mode 0: 21/9 subjects (Tables 2 and 3). Last 9 subjects -> test. For the first 21,
                the first 14 samples of each activity -> train, the remaining samples -> val.
        Mode 1: first 20 subjects -> train, last 10 subjects -> val (fold 1, Tables 4 and 5)
        Mode 2: last 20 subjects -> train, first 10 subjects -> val (fold 2)
        Mode 3: 11th to 20th subjects -> val, all other subjects -> train (fold 3)
    Scenes 2, 3, 4: ALL subjects -> test
    """
    _clear_existing_output(dst_root)
    split_name_path = os.path.join(dst_root, SPLIT_NAME_FILE)
    if os.path.exists(split_name_path):
        os.remove(split_name_path)  # Only a completed split gets a name
    _ensure_dirs(dst_root)

    for scene_id in [1]:
        scene_root = os.path.join(raw_root, f"Scene{scene_id}")
        stems = _list_scene_samples(scene_root)
        if not stems:
            print(f"[Warning] No valid samples found in: {scene_root}")
            continue
        subject_to_stems = defaultdict(list)
        for s in stems:
            parsed = _parse_filename(s)
            if parsed is None:
                continue
            subject, _, _ = parsed
            subject_to_stems[subject].append(s)
        subjects_sorted = sorted(subject_to_stems.keys())
        train_subjects = set()
        val_subjects = set()
        test_subjects = set()
        if mode == 0:
            test_subjects = set(subjects_sorted[21:])
        elif mode == 1:
            train_subjects = set(subjects_sorted[:20])
            val_subjects = set(subjects_sorted[20:])
        elif mode == 2:
            val_subjects = set(subjects_sorted[:10])
            train_subjects = set(subjects_sorted[10:])
        elif mode == 3:
            val_subjects = set(subjects_sorted[10:20])
            train_subjects = set(subjects_sorted[:10] + subjects_sorted[20:])
        train_stems = set()
        val_stems = set()
        test_stems = set()
        for subj in train_subjects:
            train_stems.update(subject_to_stems[subj])
        for subj in val_subjects:
            val_stems.update(subject_to_stems[subj])
        for subj in test_subjects:
            test_stems.update(subject_to_stems[subj])
        if mode == 0:
            for subj in subjects_sorted[:21]:
                by_activity = defaultdict(list)
                for stem in subject_to_stems[subj]:
                    _, activity, sample_idx = _parse_filename(stem)
                    by_activity[activity].append((sample_idx, stem))
                for samples in by_activity.values():
                    ordered = [stem for _, stem in sorted(samples)]
                    train_stems.update(ordered[:TRAIN_SAMPLES_PER_ACTIVITY])
                    val_stems.update(ordered[TRAIN_SAMPLES_PER_ACTIVITY:])
                train_subjects.add(subj)
                val_subjects.add(subj)
        n = len(stems)
        print(f"\n[Scene {scene_id}] total samples: {n}")
        print(f"  Train subjects: {sorted(train_subjects)}")
        print(f"  Val subjects: {sorted(val_subjects)}")
        print(f"  Test subjects: {sorted(test_subjects)}")
        print(f"  Train samples: {len(train_stems)}")
        print(f"  Val samples: {len(val_stems)}")
        print(f"  Test samples: {len(test_stems)}")
        _clear_scene_dirs(dst_root, scene_id)
        for stem in tqdm(stems, desc=f"Splitting Scene{scene_id}"):
            if stem in train_stems:
                split_name = "train_data"
            elif stem in val_stems:
                split_name = "val_data"
            elif stem in test_stems:
                split_name = "test_data"
            else:
                continue
            _link_modalities(scene_root, dst_root, split_name, scene_id, stem)
    for scene_id in [2, 3, 4]:
        scene_root = os.path.join(raw_root, f"Scene{scene_id}")
        stems = _list_scene_samples(scene_root)
        if not stems:
            print(f"[Warning] No valid samples found in: {scene_root}")
            continue
        print(f"\n[Scene {scene_id}] total samples: {len(stems)}")
        print("  All samples go to test_data.")
        _clear_scene_dirs(dst_root, scene_id)
        for stem in tqdm(stems, desc=f"Splitting Scene{scene_id}"):
            _link_modalities(scene_root, dst_root, "test_data", scene_id, stem)
    print("\n[Done] Split completed.")
    print(f"Output root: {dst_root}")
    _print_directory_sample_counts(dst_root)
    _print_unique_subjects_by_directory(dst_root)

    generate_txt_mix(dst_root, "dml")
    with open(split_name_path, 'w') as f:
        f.write(f"split{mode}\n")
    print(f"\nActive split: split{mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split XRF55 by subject (scenes 2-4 always go to test_data)')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Scene 1 split. 0: 21/9 subjects (last 9 test; first 21: 14 samples per activity train, rest val). '
                             '1: first 20 train, last 10 val. 2: last 20 train, first 10 val. '
                             '3: 11th-20th val, others train.')
    args = parser.parse_args()
    split_by_subject_across_scenes(mode=args.mode)
