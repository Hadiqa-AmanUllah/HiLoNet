"""Generate label files from the split folders created by split_train_test.py."""

import os


def generate_txt_mix(data_src_path, txt_name):
    """Generate train/val/test/finetune txt from processed split folders.

    Supports both legacy flat layout:
      train_data/RFID/*.npy
      test_data/RFID/*.npy

        and scene-aware layout produced by split_train_test.py:
      train_data/scene1/RFID/*.npy
            val_data/scene1/RFID/*.npy
      test_data/scene1/RFID/*.npy
      finetune_data/scene1/RFID/*.npy

    For scene-aware layout, the stored filename becomes `sceneX/stem` so the
    dataset loaders can resolve samples unambiguously across scenes.
    """
    train_list = []
    val_list = []
    test_list = []

    def collect_split_entries(split_name):
        split_root = os.path.join(data_src_path, split_name)
        entries = []
        flat_rfid_dir = os.path.join(split_root, "RFID")
        if not os.path.isdir(split_root):
            return entries

        scene_dirs = [d for d in sorted(os.listdir(split_root)) if d.lower().startswith('scene')]
        has_scene_layout = any(os.path.isdir(os.path.join(split_root, d, 'RFID')) for d in scene_dirs)

        if has_scene_layout:
            for scene_dir in scene_dirs:
                scene_rfid_dir = os.path.join(split_root, scene_dir, 'RFID')
                if not os.path.isdir(scene_rfid_dir):
                    continue
                for file in sorted(os.listdir(scene_rfid_dir)):
                    if not file.endswith('.npy'):
                        continue
                    stem = file.split('.')[0]
                    parts = stem.split('_')
                    if len(parts) < 2:
                        continue
                    vol_index = parts[0]
                    act_index = parts[1]
                    filename = f"{scene_dir}/{stem}"
                    entries.append(f"{filename},{vol_index},{act_index}\n")
            return entries

        if os.path.isdir(flat_rfid_dir):
            for file in sorted(os.listdir(flat_rfid_dir)):
                if not file.endswith('.npy'):
                    continue
                filename = file.split('.')[0]
                parts = filename.split('_')
                if len(parts) < 2:
                    continue
                vol_index = parts[0]
                act_index = parts[1]
                entries.append(f"{filename},{vol_index},{act_index}\n")
            return entries
        return entries

    train_list = collect_split_entries('train_data')
    val_list = collect_split_entries('val_data')
    test_list = collect_split_entries('test_data')
    if not val_list and test_list:
        print("[Info] val_data is empty, falling back to test_data for *_val labels.")
        val_list = list(test_list)
    finetune_list = collect_split_entries('finetune_data')

    def write_list(file_path, entries, tag):
        with open(file_path, "w") as f:
            for item in entries:
                f.write(item)
        print(f"{tag} len: {len(entries)}")

    def group_by_scene(entries):
        """Group lines by scene prefix from filename field: sceneX/stem,..."""
        grouped = {}
        for line in entries:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            filename = parts[0]
            if '/' in filename:
                scene_name = filename.split('/', 1)[0]
            else:
                # Legacy flat layout has no scene prefix; skip scene grouping.
                continue
            grouped.setdefault(scene_name, []).append(line + "\n")
        return grouped

    label_dir = data_src_path
    os.makedirs(label_dir, exist_ok=True)
    write_list(os.path.join(label_dir, txt_name + "_train.txt"), train_list, "train_dataset")
    write_list(os.path.join(label_dir, txt_name + "_val.txt"), val_list, "val_dataset")
    write_list(os.path.join(label_dir, txt_name + "_test.txt"), test_list, "test_dataset")
    write_list(os.path.join(label_dir, txt_name + "_finetune.txt"), finetune_list, "finetune_dataset")

    train_by_scene = group_by_scene(train_list)
    val_by_scene = group_by_scene(val_list)
    test_by_scene = group_by_scene(test_list)
    finetune_by_scene = group_by_scene(finetune_list)

    all_scene_names = sorted(
        set(train_by_scene.keys())
        | set(val_by_scene.keys())
        | set(test_by_scene.keys())
        | set(finetune_by_scene.keys())
    )
    for scene_name in all_scene_names:
        write_list(
            os.path.join(label_dir, f"{scene_name}_train.txt"),
            train_by_scene.get(scene_name, []),
            f"{scene_name}_train",
        )
        write_list(
            os.path.join(label_dir, f"{scene_name}_val.txt"),
            val_by_scene.get(scene_name, []),
            f"{scene_name}_val",
        )
        write_list(
            os.path.join(label_dir, f"{scene_name}_test.txt"),
            test_by_scene.get(scene_name, []),
            f"{scene_name}_test",
        )
        write_list(
            os.path.join(label_dir, f"{scene_name}_finetune.txt"),
            finetune_by_scene.get(scene_name, []),
            f"{scene_name}_finetune",
        )


if __name__ == '__main__':
    from split_train_test import SPLIT_OUTPUT_ROOT
    generate_txt_mix(SPLIT_OUTPUT_ROOT, "dml")
