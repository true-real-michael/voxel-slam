import sys
import time
from typing import List, Dict

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.append("..")
sys.path.append("../octreelib")
from slam.utils import KittiReader
from util import create_configuration, PipelineConfiguration, read_patch, PipelineDurations

N_ITERATIONS = 1  # number of iterations for each number of poses
N_POSES = [10, 20, 30, 50, 80, 100][::-1]  # each item cannot be larger than 100
BATCH_SIZE = 100  # number of poses processed in one batch during CUDA RANSAC

DATASET_PATH = "../evaluation/kitti"
DATASET_READER = KittiReader
USE_CUDA = True


def run_pipeline(point_clouds: List[o3d.geometry.PointCloud],
                 pipeline: PipelineConfiguration,
                 cuda: bool = False) -> PipelineDurations:
    """
    Runs pipeline with time benchmarking
    """

    middle_pose_number = len(point_clouds) // 2

    initialization_start = time.perf_counter()
    pipeline.grid.insert_points(
        middle_pose_number,
        np.asarray(point_clouds[middle_pose_number].points),
    )
    initialization_end = time.perf_counter() - initialization_start

    subdividers_start = time.perf_counter()
    pipeline.grid.subdivide(pipeline.subdividers)
    subdividers_end = time.perf_counter() - subdividers_start

    distribution_start = time.perf_counter()
    for pose_number, point_cloud in enumerate(point_clouds):
        if pose_number == middle_pose_number:
            continue
        pipeline.grid.insert_points(pose_number, np.asarray(point_cloud.points))
    distribution_end = time.perf_counter() - distribution_start

    segmenters_start = time.perf_counter()
    if cuda:
        pipeline.grid.map_leaf_points_cuda_ransac(poses_per_batch=BATCH_SIZE)
    else:
        for segmenter in pipeline.segmenters:
            pipeline.grid.map_leaf_points(segmenter)
    segmenters_end = time.perf_counter() - segmenters_start

    backend_start = time.perf_counter()
    backend_end = time.perf_counter() - backend_start

    return PipelineDurations(
        initialization=initialization_end,
        subdividers=subdividers_end,
        distribution=distribution_end,
        segmenters=segmenters_end,
        backend=backend_end,
    )


def measure(cuda, poses, iterations) -> Dict[str, float]:
    initializations_timestamps = []
    subdividers_timestamps = []
    distributions_timestamps = []
    segmenters_timestamps = []

    for sample in range(iterations):
        point_clouds = read_patch(DATASET_READER, DATASET_PATH, 0, poses)

        pipeline_durations = run_pipeline(point_clouds, create_configuration(), cuda)

        initializations_timestamps.append(pipeline_durations.initialization)
        subdividers_timestamps.append(pipeline_durations.subdividers)
        distributions_timestamps.append(pipeline_durations.distribution)
        segmenters_timestamps.append(pipeline_durations.segmenters)

    device = 'CUDA' if cuda else 'CPU'
    result = {
        'initializations': (np.mean(initializations_timestamps), np.std(initializations_timestamps)),
        'subdividers': (np.mean(subdividers_timestamps), np.std(subdividers_timestamps)),
        'distributions': (np.mean(distributions_timestamps), np.std(distributions_timestamps)),
        f'{device}_segmenters': (np.mean(segmenters_timestamps), np.std(segmenters_timestamps)),
    }

    return result


benchmark_results = {}
for poses in N_POSES:
    benchmark_results[poses] = {}
    benchmark_results[poses] |= measure(USE_CUDA, poses, N_ITERATIONS)

df = pd.DataFrame(benchmark_results).T
print(df)
df.to_csv(f'benchmark_{"CUDA" if USE_CUDA else "CPU"}.csv')
