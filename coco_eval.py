# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.utils import boolean_string

import cv2

from consts import NO_MODEL_OUTPUT_ERROR

from predictor import CocoPredictor


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--project",
        type=str,
        default="projects/coco.yml",
        help="project filepath that contains parameters",
    )
    ap.add_argument(
        "-dd",
        "--data_directory",
        type=str,
        default="datasets",
        help="data directory with datasets",
    )
    ap.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="output filepath with evaluation results",
    )
    ap.add_argument(
        "-c",
        "--compound_coef",
        type=int,
        default=0,
        help="coefficients of efficientdet",
    )
    ap.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="/path/to/weights/efficientdet.pth",
    )
    ap.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help="nms threshold, don't change it if not for testing purposes",
    )
    ap.add_argument("--cuda", type=boolean_string, default=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--float16", type=boolean_string, default=False)
    ap.add_argument(
        "--override",
        type=boolean_string,
        default=True,
        help="override previous bbox results file if exists",
    )
    return ap.parse_args()


def evaluate_coco(
    images_dir: str,
    image_ids,
    coco,
    result_file: str,
    predictor: CocoPredictor,
    iou_threshold,
    threshold=0.05,
):
    assert os.path.exists(images_dir)

    results = []

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = images_dir + image_info["file_name"]

        rois, class_ids, scores = predictor.predict(
            image=cv2.imread(image_path),
            threshold=threshold,
            iou_threshold=iou_threshold,
        )

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    "image_id": image_id,
                    "category_id": label + 1,
                    "score": float(score),
                    "bbox": box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise print(NO_MODEL_OUTPUT_ERROR)

    # write output
    if os.path.exists(result_file):
        os.remove(result_file)
    json.dump(results, open(result_file, "w"), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print("BBox")
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def run(args):
    project_name = os.path.splitext(os.path.basename(args.project))[0]

    override_prev_results = args.override
    result_file = args.output_file
    params = yaml.safe_load(open(args.project))
    obj_list = params["obj_list"]
    SET_NAME = params["val_set"]

    if not override_prev_results and os.path.exists(result_file):
        raise ValueError(
            f"Result file: {result_file} already exists, please choose another name or set `override` param to True."
        )

    data_directory = args.data_directory
    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    use_cuda = args.cuda
    gpu = args.device
    use_float16 = args.float16

    weights_path = (
        f"weights/efficientdet-d{compound_coef}.pth"
        if args.weights is None
        else args.weights
    )

    print(
        f"running coco-style evaluation on project {project_name}, weights {weights_path}..."
    )

    VAL_GT = f'{data_directory}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{data_directory}/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)

    image_ids = coco_gt.getImgIds()
    if len(image_ids) > MAX_IMAGES:
        print(
            f"Exceed {MAX_IMAGES} images in val, taking only first {MAX_IMAGES} images."
        )
    image_ids = image_ids[:MAX_IMAGES]

    predictor = CocoPredictor(
        model_path=weights_path,
        compound_coef=compound_coef,
        device="cuda" if use_cuda else "cpu",
        gpu=gpu,
        classes_number=len(obj_list),
        anchor_ratios=eval(params["anchors_ratios"]),
        anchor_scales=eval(params["anchors_scales"]),
        use_float16=use_float16,
    )

    evaluate_coco(
        images_dir=VAL_IMGS,
        image_ids=image_ids,
        coco=coco_gt,
        result_file=result_file,
        predictor=predictor,
        iou_threshold=nms_threshold,
    )

    _eval(coco_gt, image_ids, result_file)


if __name__ == "__main__":
    args = get_args()
    run(args)
