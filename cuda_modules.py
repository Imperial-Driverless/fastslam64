from pathlib import Path
from typing import Any
from pycuda.compiler import SourceModule
import pycuda.autoinit # type: ignore

class CudaModules:
    def __init__(self, PARTICLE_SIZE: int, N_PARTICLES: int, THREADS: int):  
        self.options = [
            f'-DPARTICLE_SIZE={PARTICLE_SIZE}',
            f'-DN_PARTICLES={N_PARTICLES}',
            f'-DTHREADS={THREADS}',
        ]

        module = SourceModule(Path("cuda/fastslam.cu").read_text(), options=self.options, no_extern_c=True)
        self.module = module

    def init_rng(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("init_rng")(*args, **kwargs) # type: ignore
    def write_to_c(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("write_to_c")(*args, **kwargs) # type: ignore
    def reset_weights(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("reset_weights")(*args, **kwargs) # type: ignore
    def write_(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("write_")(*args, **kwargs) # type: ignore
    def get_weights(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("get_weights")(*args, **kwargs) # type: ignore
    def predict_from_imu(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("predict_from_imu")(*args, **kwargs) # type: ignore
    def update(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("update")(*args, **kwargs) # type: ignore
    def sum_weights(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("sum_weights")(*args, **kwargs) # type: ignore
    def divide_weights(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("divide_weights")(*args, **kwargs) # type: ignore
    def get_mean_position(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("get_mean_position")(*args, **kwargs) # type: ignore
    def systematic_resample(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("systematic_resample")(*args, **kwargs) # type: ignore
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("reset")(*args, **kwargs) # type: ignore
    def prepermute(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("prepermute")(*args, **kwargs) # type: ignore
    def permute(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("permute")(*args, **kwargs) # type: ignore
    def copy_inplace(self, *args: Any, **kwargs: Any) -> None:
        return self.module.get_function("copy_inplace")(*args, **kwargs) # type: ignore
