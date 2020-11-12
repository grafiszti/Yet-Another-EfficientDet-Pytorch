import cv2
import argparse
import os

from coco_eval import run as eval_run
from train import run as train_run
import json
import shutil

from glob import glob

from predictor import CocoPredictor

test_dir = "test"
test_evaluation_result_filepath = test_dir + "/test_evaluation_result.json"
train_result_directory = test_dir + "/test_logs/"
trained_test_model = (
    train_result_directory + "test_coco_dataset/efficientdet-d0_0_2.pth"
)
test_project_yml = test_dir + "/test_coco_dataset.yml"
base_efficitnetdet_0_model = "weights/efficientdet-d0.pth"


class TestModelTraining:
    def test_training_efficient_det(self):
        train_run(
            argparse.Namespace(
                project=test_project_yml,
                compound_coef=0,
                num_workers=12,
                batch_size=1,
                head_only=True,
                lr=0.01,
                optim="adam",
                num_epochs=1,
                val_interval=1,
                save_interval=1,
                es_min_delta=0.0,
                es_patience=0,
                data_path=test_dir,
                log_path=train_result_directory,
                load_weights=base_efficitnetdet_0_model,
                saved_path=train_result_directory,
                debug=False,
            )
        )

        assert os.path.exists(train_result_directory)
        assert len(glob(os.path.dirname(trained_test_model) + "/*.pth")) > 0


class TestModelEvaluation:
    @classmethod
    def setup_class(cls):
        train_run(
            argparse.Namespace(
                project=test_project_yml,
                compound_coef=0,
                num_workers=12,
                batch_size=1,
                head_only=True,
                lr=0.01,
                optim="adam",
                num_epochs=1,
                val_interval=1,
                save_interval=1,
                es_min_delta=0.0,
                es_patience=0,
                data_path=test_dir,
                log_path=train_result_directory,
                load_weights=base_efficitnetdet_0_model,
                saved_path=train_result_directory,
                debug=False,
            )
        )

    def test_evaluation(self):
        predictor = CocoPredictor(
            model_path=base_efficitnetdet_0_model,
            compound_coef=0,
            device="cpu",
            gpu=0,
            classes_number=1,
            anchor_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
            anchor_scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
            use_float16=False,
        )

        rois, class_ids, scores = predictor.predict(
            image=cv2.imread("test/img.png"), threshold=0.2,
        )

        assert rois.shape[0] > 0
        assert class_ids.shape[0] > 0
        assert scores.shape[0] > 0

    def test_coco_eval(self):
        eval_run(
            argparse.Namespace(
                project=test_project_yml,
                data_directory=test_dir,
                weights=trained_test_model,
                compound_coef=0,
                output_file=test_evaluation_result_filepath,
                override=True,
                nms_threshold=0.5,
                cuda=False,
                device=0,
                float16=False,
            )
        )

        assert os.path.exists(test_evaluation_result_filepath)

        with open(test_evaluation_result_filepath, "r") as f:
            data = json.load(f)

        assert len(data) > 0
        assert set([a["image_id"] for a in data]) == {3}
        assert set([a["category_id"] for a in data]) == {1}


def teardown_module():
    for path in [test_evaluation_result_filepath, train_result_directory]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"Removing: {path}")
