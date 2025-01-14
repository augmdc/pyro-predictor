#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import uuid
import base64
import numpy as np
import cv2
import imgviz

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    exit(1)


def decode_base64_mask(encoded_mask, img_shape, points):
    try:
        mask_data = base64.b64decode(encoded_mask)
        mask_array = np.frombuffer(mask_data, dtype=np.uint8)
        small_mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
        if small_mask is None:
            raise ValueError("Decoded mask is None.")

        x1, y1 = int(points[0][0]), int(points[0][1])
        x2, y2 = int(points[1][0]), int(points[1][1])

        expected_width = x2 - x1 + 1
        expected_height = y2 - y1 + 1

        if (small_mask.shape[1] != expected_width) or (
            small_mask.shape[0] != expected_height
        ):
            raise ValueError(
                f"Mask dimensions {small_mask.shape} do not match bounding box dimensions ({expected_height}, {expected_width})."
            )

        full_mask = np.zeros(img_shape[:2], dtype=np.uint8)
        full_mask[y1 : y2 + 1, x1 : x2 + 1] = small_mask
        return full_mask
    except Exception as e:
        print(f"Error decoding mask: {e}")
        return None


def create_binary_mask(points, img_shape):
    polygon = np.array(points, dtype=np.int32)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    parser.add_argument(
        "--use-polygons", help="use polygons for segmentation", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[],
        type="instances",
        annotations=[],
        categories=[],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i + 1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        with open(filename, "r") as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img_path = osp.join(osp.dirname(filename), label_data["imagePath"])
        img = cv2.imread(img_path)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=label_data["imageHeight"],
                width=label_data["imageWidth"],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}
        for shape in label_data["shapes"]:
            label = shape["label"]
            group_id = shape.get("group_id", uuid.uuid1())
            img_shape = (label_data["imageHeight"], label_data["imageWidth"])

            if "mask" in shape:  # Handle Base64-encoded mask
                mask = decode_base64_mask(
                    shape["mask"],
                    img_shape,
                    shape["points"],
                )
                if mask is None:
                    print(
                        f"[Error] Skipping invalid mask for label '{label}' in file: {filename}"
                    )
                    continue
            else:
                mask = create_binary_mask(shape["points"], img_shape)

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            encoded_mask = pycocotools.mask.encode(np.asfortranarray(mask))
            for k, v in encoded_mask.items():
                if isinstance(v, bytes):
                    encoded_mask[k] = v.decode("utf-8")
            area = float(pycocotools.mask.area(encoded_mask))
            bbox = pycocotools.mask.toBbox(encoded_mask).flatten().tolist()

            segmentation = (
                encoded_mask
                if not args.use_polygons
                else [list(np.array(shape["points"]).flatten())]
            )

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentation,
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            viz = imgviz.instances2rgb(
                image=img,
                labels=[class_name_to_id[cls_name] for cls_name, _ in masks.keys()],
                masks=[mask.astype(bool) for _, mask in masks.items()],
                captions=[cls_name for cls_name, _ in masks.keys()],
                font_size=15,
            )
            out_viz_file = osp.join(args.output_dir, "Visualization", base + ".jpg")
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
