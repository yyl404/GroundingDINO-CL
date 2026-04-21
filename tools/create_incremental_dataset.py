"""Create class-incremental dataset by splitting source dataset into multiple tasks.

Usage:
    $ python tools/create_incremental_dataset.py \\
        --source_cfg <path/to/source_dataset.yaml> \\
        --output_dir <path/to/output_dir> \\
        --n_classes <n1> <n2> ... \\
        --split <split1> <split2> ... (optional)

    Arguments:
        --source_cfg: Path to the source dataset configuration file (.yaml file)
        --output_dir: Path to the output directory where incremental dataset will be created
        --n_classes: Number of classes for each task (space-separated list, e.g., "15 5" for two tasks)
        --split: Dataset splits to create (default: "train val test")
        
Examples:
    # Create incremental dataset with 15 classes in task 1 and 5 classes in task 2
    $ python tools/create_incremental_dataset.py \\
        --source_cfg data/VOC-YOLO/VOC.yaml \\
        --output_dir data/VOC_inc_15_5 \\
        --n_classes 15 5

    # Create incremental dataset with 4 tasks, each with 5 classes
    $ python tools/create_incremental_dataset.py \\
        --source_cfg data/VOC-YOLO/VOC.yaml \\
        --output_dir data/VOC_inc_5x4 \\
        --n_classes 5 5 5 5

Three dataset splitting modes are supported:

1. full-split: Different stages have completely isolated datasets with non-overlapping image sets.
   When traversing the original dataset, images are preferentially assigned to the stage with the smallest
   (num_images / num_classes) ratio that has an intersection with the classes appearing in the current image.

2. sample-filter: Each stage's dataset contains and only contains all images that contain instances of that stage's classes.

3. label-filter: All stages share the same set of images, only filtering the class labels on the images.

The task-specified classes are split in the order of class id in whole dataset.
"""

import argparse
import sys
import traceback
import os
import shutil
import glob
import warnings
from tqdm import tqdm
import yaml


SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg', 'bmp']
SUPPORTED_LABEL_EXTENSIONS = ['.txt']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create class-incremental dataset")
    parser.add_argument("--source_cfg", type=str, required=True, help="The path to the source dataset's config file")
    parser.add_argument("--output_dir", type=str, required=True, help="The path to create output dataset's directory")
    parser.add_argument("--n_classes", type=int, nargs='+', required=True, help="The number of classes for each task")
    parser.add_argument("--split", type=str, default=['train', 'val', 'test'], nargs='+', help="The splits to create the \
        dataset for, default is \"train val test\"")
    parser.add_argument("--mode", type=str, choices=['full-split', 'sample-filter', 'label-filter'], default='sample-filter',
                        help="Dataset splitting mode: 'full-split' (isolated image sets), 'sample-filter' (filter by image), 'label-filter' (filter by label)")
    args = parser.parse_args()

    # load source dataset classes
    with open(args.source_cfg, "r", encoding="utf-8") as f:
        source_dataset_yaml = yaml.safe_load(f)
    source_classes = source_dataset_yaml["names"]

    # check if the source dataset has corresponding splits
    splits = []
    for _split in args.split:
        if _split in source_dataset_yaml.keys():
            splits.append(_split)
        else:
            warnings.warn(f"Source dataset config file {args.source_cfg} does not have corresponding split {_split}, \
                skipping...")

    # check if the output directory exists
    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists, remove it or not? (Yes/No/Cancel)")
        answer = input()
        if answer.lower() == "yes" or answer.lower() == "y":
            shutil.rmtree(args.output_dir)
            print(f"Output directory {args.output_dir} removed.")
        elif answer.lower() == "cancel" or answer.lower() == "c":
            print("Aborting...")
            sys.exit(1)

    task_classes = []
    classes_id_map_source2task = []
    # split the source dataset classes by provided n_classes
    # Get sorted class ids to ensure consistent ordering
    sorted_source_class_ids = sorted(source_classes.keys())
    for t, n_classes in enumerate(args.n_classes):
        task_classes.append({})
        classes_id_map_source2task.append({}) # map from source class id to task class id
        for i in range(n_classes):
            source_class_id = sorted_source_class_ids[sum(args.n_classes[:t]) + i]
            class_name = source_classes[source_class_id]
            task_classes[t][i] = class_name
            classes_id_map_source2task[t][source_class_id] = i

    # initialize the cumulative task classes and classes id map
    task_cumulative_classes = []
    classes_id_map_source2task_cumulative = []
    for t in range(len(task_classes)):
        task_cumulative_classes.append({})
        classes_id_map_source2task_cumulative.append({})
        class_counter = 0
        for i in range(t+1):
            # Iterate task classes in sorted order by class ID for consistency
            for task_class_id in sorted(task_classes[i].keys()):
                _class_name = task_classes[i][task_class_id]
                # Find matching source class id (search in sorted order for consistency)
                for source_class_id in sorted(source_classes.keys()):
                    source_class_name = source_classes[source_class_id]
                    if source_class_name == _class_name:
                        task_cumulative_classes[t][class_counter] = _class_name
                        classes_id_map_source2task_cumulative[t][source_class_id] = class_counter
                        class_counter += 1
                        break

    # initialize task image counts
    task_image_counts = {t: {_split: 0 for _split in splits} for t in range(len(task_classes))}
    
    # create output directory for each task
    for t in range(len(task_classes)):
        for _split in splits:
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}/images/{_split}"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}/labels/{_split}"), exist_ok=True)
            if t>0:
                os.makedirs(os.path.join(args.output_dir,
                    f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}/images/{_split}"),
                    exist_ok=True)
                os.makedirs(os.path.join(args.output_dir,
                    f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}/labels/{_split}"),
                    exist_ok=True)
    
    for _split in splits:
        if isinstance(source_dataset_yaml[_split], str):
            image_dirs = [source_dataset_yaml[_split]]
            label_dirs = [source_dataset_yaml[_split].replace("images", "labels")]
        elif isinstance(source_dataset_yaml[_split], list):
            image_dirs = source_dataset_yaml[_split]
            label_dirs = [_image_dir.replace("images", "labels") for _image_dir in image_dirs]
        else:
            raise ValueError(f"Invalid split configuration in source dataset config file \
                (must be a string or a list of strings).")
        
        # get all label files and image files in the directories
        label_files = []
        image_files = []
        for _image_dir, _label_dir in zip(image_dirs, label_dirs):
            # If the image or label directory does not exist, 
            # treat it as a relative path from the source dataset config file and try again
            if not os.path.exists(_image_dir):
                _image_dir = os.path.join(os.path.dirname(args.source_cfg), _image_dir)
            if not os.path.exists(_label_dir):
                _label_dir = os.path.join(os.path.dirname(args.source_cfg), _label_dir)
            # If the image or label directory still does not exist, raise an error
            if not os.path.exists(_image_dir):
                raise ValueError(f"Image directory {_image_dir} does not exist.")
            if not os.path.exists(_label_dir):
                raise ValueError(f"Label directory {_label_dir} does not exist.")
            # Get all image and label files in the directories
            for _img_ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.extend(glob.glob(os.path.join(_image_dir, f'*{_img_ext.lower()}')))
                image_files.extend(glob.glob(os.path.join(_image_dir, f'*{_img_ext.upper()}')))
            for _label_ext in SUPPORTED_LABEL_EXTENSIONS:
                label_files.extend(glob.glob(os.path.join(_label_dir, f'*{_label_ext.lower()}')))
                label_files.extend(glob.glob(os.path.join(_label_dir, f'*{_label_ext.upper()}')))
            
        # process each label file
        for _label_file in tqdm(label_files, desc=f"Processing {_split} split"):
            # read the label file, find out all classes in the file
            classes_in_file = set()
            try:
                with open(_label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                            class_id = int(parts[0])
                            classes_in_file.add(class_id)
            except:
                warnings.warn(f"Error reading label file {_label_file}, details as follows: \n{traceback.format_exc()}")
                continue
                
            # Get the corresponding image file path
            source_image_path = None
            source_label_path = _label_file
            for _image_file in image_files:
                if os.path.basename(_image_file).split('.')[0] == os.path.basename(_label_file).split('.')[0]:
                    source_image_path = _image_file
                    break
            if source_image_path is None or not os.path.exists(source_image_path):
                warnings.warn(f"Label file {source_label_path} corresponds to no image file, skipped")
                continue
            
            # Process image based on the selected mode
            if args.mode == 'full-split':
                # Full-split mode: assign image to one task only
                # Find tasks that have intersection with classes in this image
                candidate_tasks = []
                for t in range(len(task_classes)):
                    task_class_ids = set(classes_id_map_source2task[t].keys())
                    if classes_in_file.intersection(task_class_ids):
                        # Calculate (num_images / num_classes) ratio for this task
                        num_classes = len(task_classes[t])
                        num_images = task_image_counts[t][_split]
                        ratio = num_images / num_classes if num_classes > 0 else float('inf')
                        candidate_tasks.append((t, ratio))
                
                # Assign to task with smallest ratio
                if candidate_tasks:
                    candidate_tasks.sort(key=lambda x: x[1])
                    assigned_task = candidate_tasks[0][0]
                    
                    # Copy image and labels to assigned task
                    task_dir = os.path.join(args.output_dir, f"task_{assigned_task+1}_cls_{len(task_classes[assigned_task].values())}")
                    dest_image_path = os.path.join(task_dir, "images", _split, os.path.basename(source_image_path))
                    shutil.copy2(source_image_path, dest_image_path)
                    
                    dest_label_path = os.path.join(task_dir, "labels", _split, os.path.basename(source_label_path))
                    with open(source_label_path, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                        for i, line in enumerate(src_f):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                source_class_id = int(parts[0])
                                if source_class_id in classes_id_map_source2task[assigned_task].keys():
                                    task_class_id = classes_id_map_source2task[assigned_task][source_class_id]
                                    parts[0] = str(task_class_id)
                                    dst_f.write(' '.join(parts) + '\n')
                            else:
                                warnings.warn(f"Invalid label format in file {source_label_path} line {i+1}, skipping...")
                                continue
                    
                    task_image_counts[assigned_task][_split] += 1
                    
                    # Also add to cumulative task directories (from assigned_task onwards)
                    for end_task in range(assigned_task, len(task_classes)):
                        if end_task == 0:
                            continue
                        cumulative_class_ids = set(classes_id_map_source2task_cumulative[end_task].keys())
                        if classes_in_file.intersection(cumulative_class_ids):
                            task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{end_task+1}_cls_{len(task_cumulative_classes[end_task].values())}")
                            dest_cumulative_image_path = os.path.join(task_cumulative_dir, "images", _split, os.path.basename(source_image_path))
                            if not os.path.exists(dest_cumulative_image_path):
                                shutil.copy2(source_image_path, dest_cumulative_image_path)
                            
                            dest_cumulative_label_path = os.path.join(task_cumulative_dir, "labels", _split, os.path.basename(source_label_path))
                            with open(source_label_path, 'r') as src_f, open(dest_cumulative_label_path, 'w') as dst_f:
                                for i, line in enumerate(src_f):
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        source_class_id = int(parts[0])
                                        if source_class_id in classes_id_map_source2task_cumulative[end_task].keys():
                                            task_class_id = classes_id_map_source2task_cumulative[end_task][source_class_id]
                                            parts[0] = str(task_class_id)
                                            dst_f.write(' '.join(parts) + '\n')
                                    
            elif args.mode == 'sample-filter':
                # Sample-filter mode: each task contains all images with its classes
                for t in range(len(task_classes)):
                    task_class_ids = set(classes_id_map_source2task[t].keys())
                    if classes_in_file.intersection(task_class_ids):
                        # Copy the image file to the task directory
                        task_dir = os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}")
                        dest_image_path = os.path.join(task_dir, "images", _split, os.path.basename(source_image_path))
                        shutil.copy2(source_image_path, dest_image_path)

                        # Copy the label file and filter/convert the class ids to only show task-specified classes
                        dest_label_path = os.path.join(task_dir, "labels", _split, os.path.basename(source_label_path))
                        with open(source_label_path, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                            for i, line in enumerate(src_f):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    source_class_id = int(parts[0])
                                    if source_class_id in classes_id_map_source2task[t].keys():
                                        task_class_id = classes_id_map_source2task[t][source_class_id]
                                        parts[0] = str(task_class_id)
                                        dst_f.write(' '.join(parts) + '\n')
                                else:
                                    warnings.warn(f"Invalid label format in file {source_label_path} line {i+1}, skipping...")
                                    continue

                        task_image_counts[t][_split] += 1

                # Copy the image file to cumulative task directories if it contains instances of cumulative classes
                for end_task in range(len(task_classes)):
                    if end_task == 0:
                        continue
                    cumulative_class_ids = set(classes_id_map_source2task_cumulative[end_task].keys())
                    if classes_in_file.intersection(cumulative_class_ids):
                        task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{end_task+1}_cls_{len(task_cumulative_classes[end_task].values())}")
                        dest_cumulative_image_path = os.path.join(task_cumulative_dir, "images", _split, os.path.basename(source_image_path))
                        shutil.copy2(source_image_path, dest_cumulative_image_path)

                        dest_cumulative_label_path = os.path.join(task_cumulative_dir, "labels", _split, os.path.basename(source_label_path))
                        with open(source_label_path, 'r') as src_f, open(dest_cumulative_label_path, 'w') as dst_f:
                            for i, line in enumerate(src_f):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    source_class_id = int(parts[0])
                                    if source_class_id in classes_id_map_source2task_cumulative[end_task].keys():
                                        task_class_id = classes_id_map_source2task_cumulative[end_task][source_class_id]
                                        parts[0] = str(task_class_id)
                                        dst_f.write(' '.join(parts) + '\n')
                                        
            elif args.mode == 'label-filter':
                # Label-filter mode: all tasks share the same images, only filter labels
                # Copy image to all task directories (no condition check)
                for t in range(len(task_classes)):
                    task_dir = os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}")
                    dest_image_path = os.path.join(task_dir, "images", _split, os.path.basename(source_image_path))
                    shutil.copy2(source_image_path, dest_image_path)

                    # Filter labels to only show task-specified classes
                    dest_label_path = os.path.join(task_dir, "labels", _split, os.path.basename(source_label_path))
                    with open(source_label_path, 'r') as src_f, open(dest_label_path, 'w') as dst_f:
                        for i, line in enumerate(src_f):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                source_class_id = int(parts[0])
                                if source_class_id in classes_id_map_source2task[t].keys():
                                    task_class_id = classes_id_map_source2task[t][source_class_id]
                                    parts[0] = str(task_class_id)
                                    dst_f.write(' '.join(parts) + '\n')

                    task_image_counts[t][_split] += 1

                # Copy image to cumulative task directories (no condition check)
                for end_task in range(len(task_classes)):
                    if end_task == 0:
                        continue
                    task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{end_task+1}_cls_{len(task_cumulative_classes[end_task].values())}")
                    dest_cumulative_image_path = os.path.join(task_cumulative_dir, "images", _split, os.path.basename(source_image_path))
                    shutil.copy2(source_image_path, dest_cumulative_image_path)

                    dest_cumulative_label_path = os.path.join(task_cumulative_dir, "labels", _split, os.path.basename(source_label_path))
                    with open(source_label_path, 'r') as src_f, open(dest_cumulative_label_path, 'w') as dst_f:
                        for i, line in enumerate(src_f):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                source_class_id = int(parts[0])
                                if source_class_id in classes_id_map_source2task_cumulative[end_task].keys():
                                    task_class_id = classes_id_map_source2task_cumulative[end_task][source_class_id]
                                    parts[0] = str(task_class_id)
                                    dst_f.write(' '.join(parts) + '\n')
    
    # create the yaml config file for each task
    for t in range(len(task_classes)):
        task_dir = os.path.join(args.output_dir, f"task_{t+1}_cls_{len(task_classes[t].values())}")
        task_cumulative_dir = os.path.join(args.output_dir, f"task_1-{t+1}_cls_{len(task_cumulative_classes[t].values())}")
        
        # create the task config
        task_config = {
            'names': task_classes[t]
        }
        for _split in splits:
            task_config[_split] = f"images/{_split}"
        yaml_path = os.path.join(task_dir, 'data.yaml')
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(task_config, f, sort_keys=False, allow_unicode=True)

        # create the cumulative config
        if t > 0:
            cumulative_config = {
                'names': task_cumulative_classes[t]
            }
            for _split in splits:
                cumulative_config[_split] = f"images/{_split}"
            yaml_cumulative_path = os.path.join(task_cumulative_dir, 'data.yaml')
            with open(yaml_cumulative_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cumulative_config, f, sort_keys=False, allow_unicode=True)

        print(f"Task {t+1} completed: {len(task_classes[t].values())} classes")
        for _split in splits:
            print(f"  {_split}: {task_image_counts[t][_split]} images")
        print(f"  Task config saved to: {yaml_path}")
        if t > 0:
            print(f"  Cumulative config saved to: {yaml_cumulative_path}")