from pathlib import Path
from pycuda.compiler import SourceModule
import pycuda.autoinit # type: ignore

class CudaModules:
    def __init__(self, PARTICLE_SIZE: int, N_PARTICLES: int, THREADS: int):  
        self.options = [
            f'-DPARTICLE_SIZE={PARTICLE_SIZE}',
            f'-DN_PARTICLES={N_PARTICLES}',
            f'-DTHREADS={THREADS}',
        ]

        self.predict = self.load_module("cuda/predict.cu", no_extern_c=True)
        self.update = self.load_module("cuda/update.cu")
        self.rescale = self.load_module("cuda/rescale.cu")
        self.resample = self.load_module("cuda/resample.cu")
        self.weights_and_mean = self.load_module("cuda/weights_and_mean_position.cu")
        self.permute = self.load_module("cuda/permute.cu")

    def load_module(self, path: str, no_extern_c: bool=False) -> SourceModule:
        return SourceModule(Path(path).read_text(), options=self.options, no_extern_c=no_extern_c)
        
