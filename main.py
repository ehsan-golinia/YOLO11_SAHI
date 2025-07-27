# Import all the necessary modules
import cv2
import time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_model_weights
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device