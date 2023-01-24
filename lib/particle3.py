import numpy as np
from typing import Any, NewType

FlatParticles = NewType('FlatParticles', 'np.ndarray[Any, Any]')

class FlatParticle(object):
    @staticmethod
    def x(particles: FlatParticles):
        max_landmarks = int(particles[4])
        step = 6 + 7*max_landmarks
        return particles[0::step]

    @staticmethod
    def y(particles: FlatParticles):
        max_landmarks = int(particles[4])
        step = 6 + 7*max_landmarks
        return particles[1::step]

    @staticmethod
    def w(particles: FlatParticles):
        max_landmarks = int(particles[4])
        step = 6 + 7*max_landmarks
        return particles[3::step]

    @staticmethod
    def len(particles: FlatParticles):
        max_landmarks = int(particles[4])
        length = particles.shape[0]
        return int(length/(6 + 7*max_landmarks))

    @staticmethod
    def get_particle(particles: FlatParticles, i: int):
        max_landmarks = int(particles[4])
        size = 6 + 7*max_landmarks
        offset = size * i
        return particles[offset:offset+size]

    @staticmethod
    def get_landmarks(particles: FlatParticles, i: int):
        particle = FlatParticle.get_particle(particles, i)
        n_landmarks = int(particle[5])

        return particle[6:6+2*n_landmarks].reshape((n_landmarks, 2))

    @staticmethod
    def get_covariances(particles: FlatParticles, i: int):
        particle = FlatParticle.get_particle(particles, i)
        max_landmarks = int(particle[4])
        n_landmarks = int(particle[5])


        cov_array = particle[6+2*max_landmarks:6+6*max_landmarks]
        covariances = np.zeros((n_landmarks, 2, 2), dtype=np.float64)

        for i in range(n_landmarks):
            covariances[i, 0, 0] = cov_array[4*i]
            covariances[i, 0, 1] = cov_array[4*i + 1]
            covariances[i, 1, 0] = cov_array[4*i + 2]
            covariances[i, 1, 1] = cov_array[4*i + 3]

        return covariances

    @staticmethod
    def get_initial_particles(n_particles: int, max_landmarks: int, starting_position: 'np.ndarray[Any, Any]', sigma: float) -> FlatParticles:
        step = 6 + 7*max_landmarks
        particles = np.zeros(n_particles * step, dtype=np.float64)

        particles[0::step] = starting_position[0]
        particles[1::step] = starting_position[1]
        particles[2::step] = starting_position[2]
        particles[3::step] = 1/n_particles
        particles[4::step] = float(max_landmarks)

        return FlatParticles(particles)

    @staticmethod
    def neff(weights: 'np.ndarray[Any, Any]') -> float:
        return 1.0/np.sum(np.square(weights)) # type: ignore
