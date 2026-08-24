# This functional source module is assembled into one shared runtime.
from __future__ import annotations

from __future__ import annotations

import argparse

import fcntl

import gc

import json

import math

import os

import re

import shutil

import subprocess

import threading

import time

import uuid

import warnings

import weakref

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

from contextlib import contextmanager

from dataclasses import asdict, dataclass, replace

from pathlib import Path

for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_environment_name, "1")

import numpy as np

import cv2

import tifffile as tf

from PIL import Image

from scipy import ndimage as ndi

from skimage import exposure, filters, measure, morphology, segmentation, transform

warnings.filterwarnings("ignore", category=FutureWarning)
