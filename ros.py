from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from lib.plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from lib.particle3 import FlatParticle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit
from lib.stats import Stats
from lib.common import CUDAMemory, resample, rescale, get_pose_estimate
from cuda.fastslam import load_cuda_modules

from rclpy.node import Node

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def rb2xy(pose, rb):
    [_, _, theta] = pose
    [r, b] = rb
    return [r * np.cos(b + theta), r * np.sin(b + theta)]

def xy2rb(pose, landmark):
    position = pose[:2]
    vector_to_landmark = np.array(landmark - position, dtype=np.float64)

    r = np.linalg.norm(vector_to_landmark)
    b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]
    b = wrap_angle(b)

    return r, b

class FastSLAMNode(Node):
    def __init__(self, config: Any):
        assert config.THREADS <= 1024 # cannot run more in a single block
        assert config.N >= config.THREADS
        assert config.N % config.THREADS == 0
        rng_seed = 0

        self.config = config

        self.plot = True

        if self.plot:
            fig, self.ax = plt.subplots(2, 1, figsize=(10, 5))
            self.ax[0].axis('scaled')
            self.ax[1].axis('scaled')

        self.cuda_modules = load_cuda_modules(
            THREADS=config.THREADS,
            PARTICLE_SIZE=config.PARTICLE_SIZE,
            N_PARTICLES=config.N
        )

        self.memory = CUDAMemory(config)
        self.weights = np.zeros(config.N, dtype=np.float64)
        self.particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)

        cuda.memcpy_htod(self.memory.cov, config.sensor.COVARIANCE)
        cuda.memcpy_htod(self.memory.particles, self.particles)

        np.random.seed(0)
        self.cuda_modules["predict"].get_function("init_rng")(
            np.int32(rng_seed), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        self.stats = Stats("Loop", "Measurement")
        self.stats.add_pose(config.START_POSITION, config.START_POSITION)

    def measurement_callback(self, odometry: tuple[float, float, float], est_odometry: tuple[float, float, float], measurements_rb: list[tuple[float, float, float]]):
        self.stats.start_measuring("Loop")

        self.stats.start_measuring("Measurement")
        self.stats.stop_measuring("Measurement")

        self.cuda_modules["resample"].get_function("reset_weights")(
            self.memory.particles,
            block=(config.THREADS, 1, 1), grid=(self.config.N//config.THREADS, 1, 1)
        )

        cuda.memcpy_htod(self.memory.measurements, measurements_rb)

        self.cuda_modules["predict"].get_function("predict_from_imu")(
            self.memory.particles,
            np.float64(est_odometry[0]), np.float64(est_odometry[1]), np.float64(est_odometry[2]),
            np.float64(self.config.ODOMETRY_VARIANCE[0] ** 0.5), np.float64(self.config.ODOMETRY_VARIANCE[1] ** 0.5), np.float64(self.config.ODOMETRY_VARIANCE[2] ** 0.5),
            block=(self.config.THREADS, 1, 1), grid=(self.config.N//self.config.THREADS, 1, 1)
        )

        block_size = self.config.N if self.config.N < 32 else 32

        self.cuda_modules["update"].get_function("update")(
            self.memory.particles, np.int32(1),
            self.memory.scratchpad, np.int32(self.memory.scratchpad_block_size),
            self.memory.measurements,
            np.int32(self.config.N), np.int32(len(measurements_rb)),
            self.memory.cov, np.float64(self.config.THRESHOLD),
            np.float64(self.config.sensor.RANGE), np.float64(self.config.sensor.FOV),
            np.int32(self.config.MAX_LANDMARKS),
            block=(block_size, 1, 1), grid=(self.config.N//block_size, 1, 1)
        )

        rescale(self.cuda_modules, self.config, self.memory)
        estimate = get_pose_estimate(self.cuda_modules, self.config, self.memory)

        self.stats.add_pose([pose[0], pose[1], pose[2]], estimate)

        if self.plot:
            cuda.memcpy_dtoh(self.particles, self.memory.particles)

            self.ax[0].clear()
            self.ax[1].clear()
            self.ax[0].set_xlim([-160, 10])
            self.ax[0].set_ylim([-30, 50])
            self.ax[1].set_xlim([-160, 10])
            self.ax[1].set_ylim([-30, 50])
            self.ax[0].set_axis_off()
            self.ax[1].set_axis_off()

            plot_sensor_fov(self.ax[0], pose, self.config.sensor.RANGE, self.config.sensor.FOV)
            plot_sensor_fov(self.ax[1], pose, self.config.sensor.RANGE, self.config.sensor.FOV)

            visible_measurements = np.array([rb2xy(pose, m) for m in measurements_rb])

            if(visible_measurements.size != 0):
                plot_connections(self.ax[0], pose, visible_measurements + pose[:2])

            plot_landmarks(self.ax[0], self.config.LANDMARKS, color="blue", zorder=100)
            plot_history(self.ax[0], self.stats.ground_truth_path, color='green')
            plot_history(self.ax[0], self.stats.predicted_path, color='orange')
            # plot_history(self.ax[0], self.config.ODOMETRY[:i], color='red')

            plot_particles_weight(self.ax[0], self.particles)
            if(visible_measurements.size != 0):
                plot_measurement(self.ax[0], pose[:2], visible_measurements, color="orange", zorder=103)

            best = np.argmax(FlatParticle.w(self.particles))
            plot_landmarks(self.ax[1], self.config.LANDMARKS, color="black")
            covariances = FlatParticle.get_covariances(self.particles, best)

            plot_map(self.ax[1], FlatParticle.get_landmarks(self.particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(self.particles, best)):
                plot_confidence_ellipse(self.ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)


        self.cuda_modules["weights_and_mean"].get_function("get_weights")(
            self.memory.particles, self.memory.weights,
            block=(self.config.THREADS, 1, 1), grid=(self.config.N//self.config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(self.weights, self.memory.weights)

        neff = FlatParticle.neff(self.weights)
        if neff < 0.6*self.config.N:
            resample(self.cuda_modules, self.config, self.weights, self.memory, 0.5)

        self.stats.stop_measuring("Loop")


        

if __name__ == "__main__":
    from config_fsonline import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    slam_algo = FastSLAMNode(config)

    for i in range(slam_algo.config.ODOMETRY.shape[0]):
        pose = slam_algo.config.ODOMETRY[i]

        visible_measurements = slam_algo.config.sensor.MEASUREMENTS[i]
        visible_measurements = np.array([xy2rb(pose, m) for m in visible_measurements], dtype=np.float64)

        slam_algo.measurement_callback(pose, slam_algo.config.EST_ODOMETRY[i], visible_measurements)

    slam_algo.memory.free()
    slam_algo.stats.summary()
    print(slam_algo.stats.mean_path_deviation())


