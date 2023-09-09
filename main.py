from typing import Any
from pathlib import Path

from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

test_photos_path = 'path/to/test/set'

models = {
    'clahe_0': YOLO('models/clahe_0.pt'),
    'clahe_0_transfer': YOLO('models/clahe_0_transfer.pt'),
    'clahe_1': YOLO('models/clahe_1.pt'),
    'clahe_1_transfer': YOLO('models/clahe_1_transfer.pt'),
    'clahe_2': YOLO('models/clahe_2.pt'),
    'clahe_2_transfer': YOLO('models/clahe_2_transfer.pt'),
    'clahe_3': YOLO('models/clahe_3.pt'),
    'clahe_3_transfer': YOLO('models/clahe_3_transfer.pt'),
    'clahe_green_0': YOLO('models/clahe_green_0.pt'),
    'clahe_green_0_transfer': YOLO('models/clahe_green_0_transfer.pt'),
    'clahe_green_1': YOLO('models/clahe_green_1.pt'),
    'clahe_green_1_transfer': YOLO('models/clahe_green_1_transfer.pt'),
    'clahe_green_2': YOLO('models/clahe_green_2.pt'),
    'clahe_green_2_transfer': YOLO('models/clahe_green_2_transfer.pt'),
    'clahe_green_3': YOLO('models/clahe_green_3.pt'),
    'clahe_green_3_transfer': YOLO('models/clahe_green_3_transfer.pt'),
}


def correct_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def preprocess_image_to_green(img_path: str) -> None:
    img = cv2.imread(img_path)
    img = correct_contrast(img)[..., 1]

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    img = correct_contrast(img)

    return img


def serialize_box(box: Any) -> Any:
    score = {}

    score['cls'] = box.cls[0].tolist()
    score['conf'] = box.conf[0].tolist()
    score['xyxy'] = box.xyxy[0].tolist()

    return score


def serialize_pred(pred: Any, modal: Any) -> dict:
    results = []

    for result in pred:
        for box in result.boxes:
            results.append(serialize_box(box))

    return {'boxes': results, 'modality': modal}


def get_get_img_prediction(img_path: str) -> dict:
    img_clahe = preprocess_image(img_path)
    img_clahe_green = preprocess_image_to_green(img_path)

    results = {'img_path': Path(img_path).stem}

    for model_name, model in models.items():
        results_clahe = model(img_clahe, verbose=False)
        results_clahe_green = model(img_clahe_green, verbose=False)

        serialized_pred_clahe = serialize_pred(results_clahe, 'clahe')
        serialized_pred_clahe_green = serialize_pred(results_clahe_green, 'clahe_green')

        results[model_name] = [serialized_pred_clahe, serialized_pred_clahe_green]

    return results


if __name__ == '__main__':
    for path in list(tqdm(Path(test_photos_path).glob('*.jpg'))):
        prediction = get_get_img_prediction(str(path))

        print(f'{path}: {prediction}')
