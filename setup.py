import json
from config import *
from tqdm import tqdm
from PIL import Image


def convert_bdd_to_yolo(bdd_json, img_dir, output_dir, category_mapping):
    """Converts the BDD100k json file into text labels"""
    os.makedirs(output_dir, exist_ok=True)

    with open(bdd_json) as f:
        data = json.load(f)

    for item in tqdm(data):
        file_name = item["name"]
        stem = os.path.splitext(file_name)[0]
        img_path = os.path.join(img_dir, file_name)
        label_path = os.path.join(output_dir, stem + ".txt")

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except:
            print(f"Missing image {file_name}, skipping.")
            continue

        lines = []
        for label in item.get("labels", []):
            cat = label.get("category")
            if cat not in category_mapping:
                continue
            box2d = label.get("box2d")
            if not box2d:
                continue

            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2, y1 + bh / 2
            cx, cy, bw, bh = cx / w, cy / h, bw / w, bh / h

            cls_id = category_mapping[cat]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    return


def generate_yaml_config(category_mapping, output_path="bdd100k.yaml"):
    """Generates YAML configuration for training on the BDD100k Dataset"""
    yaml_content = ""
    

    yaml_content = f"""
path: data/assignment_data_bdd/bdd100k_images_100k/bdd100k
train: images/100k/train
val: images/100k/val
test: images/100k/test

names:\n"""
    sorted_classes = sorted(category_mapping.items(), key=lambda x: x[1])
    for class_name, class_id in sorted_classes:
        yaml_content += f"  {class_id}: {class_name}\n"
    
    class_names = [class_name for class_name, _ in sorted_classes]
    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"YAML config generated: {output_path}")


def main():
    """Generate minimal text label files and JSONL files for efficient retrieval and training."""

    # generate jsonl files for faster data retrieval
    categories_list = []
    for label_file, new_label_file in zip(
        [TRAIN_LABEL_FILE, VAL_LABEL_FILE], [NEW_TRAIN_LABEL_FILE, NEW_VAL_LABEL_FILE]
    ):

        with open(label_file, "r") as f:
            content = json.load(f)

        with open(new_label_file, "w") as jsonl_output:
            for entry in tqdm(content):
                for gt in entry["labels"]:
                    categories_list.append(gt["category"])
                json.dump(entry, jsonl_output)
                jsonl_output.write("\n")

    # create class mapping json
    category_mapping = {
        category: idx for idx, category in enumerate(sorted(list(set(categories_list))))
    }

    # save mapping in a folder
    with open("data/metadata/category_map.json", "w") as f:
        json.dump(category_mapping, f)

    # convert train and val annotations into text files
    convert_bdd_to_yolo(
        bdd_json=TRAIN_LABEL_FILE,
        img_dir=os.path.join(IMAGES_DIR, "train"),
        output_dir=os.path.join(NEW_LABELS_DIR, "train"),
        category_mapping=category_mapping,
    )
    convert_bdd_to_yolo(
        bdd_json=VAL_LABEL_FILE,
        img_dir=os.path.join(IMAGES_DIR, "val"),
        output_dir=os.path.join(NEW_LABELS_DIR, "val"),
        category_mapping=category_mapping,
    )

    # create bdd100k yaml file
    generate_yaml_config(category_mapping=category_mapping)

    file_path = NEW_TRAIN_LABEL_FILE

    with open(file_path, "r") as file:
        content = file.readlines()

    # generate field mapping for image level attributes
    img_attr_dict = {}
    for line in content:
        annot_obj = json.loads(line.strip())
        for img_attr, img_attr_val in annot_obj['attributes'].items():
            if img_attr not in img_attr_dict.keys():
                img_attr_dict[img_attr] = [img_attr_val]
            else:
                if img_attr_val not in img_attr_dict[img_attr]:
                    img_attr_dict[img_attr].append(img_attr_val)

    # generate field mapping for label level attributes
    label_attr_dict = {}
    for line in content:
        annot_obj = json.loads(line.strip())
        annot_labels = annot_obj['labels']
        for label_obj in annot_labels:
            for label_attr, label_attr_val in label_obj['attributes'].items():
                if label_attr not in label_attr_dict.keys():
                    label_attr_dict[label_attr] = [label_attr_val]
                else:
                    if label_attr_val not in label_attr_dict[label_attr]:
                        label_attr_dict[label_attr].append(label_attr_val)

    # save both dictionaries in metadata
    with open("data/metadata/img_attr_dict.json", "w") as f:
        json.dump(img_attr_dict, f)
    with open("data/metadata/label_attr_dict.json", "w") as f:
        json.dump(label_attr_dict, f)

if __name__ == "__main__":
    main()
