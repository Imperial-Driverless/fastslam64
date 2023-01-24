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

        module = SourceModule(Path("cuda/fastslam.cu").read_text(), options=self.options, no_extern_c=True)
        
        self.init_rng = module.get_function("init_rng")
        self.write_to_c = module.get_function("write_to_c")
        self.reset_weights = module.get_function("reset_weights")
        self.write_ = module.get_function("reset_weights")
        self.get_weights = module.get_function("get_weights")
        self.predict_from_imu = module.get_function("predict_from_imu")
        self.update = module.get_function("update")
        self.sum_weights = module.get_function("sum_weights")
        self.divide_weights = module.get_function("divide_weights")
        self.get_mean_position = module.get_function("get_mean_position")
        self.systematic_resample = module.get_function("systematic_resample")
        self.reset = module.get_function("reset")
        self.prepermute = module.get_function("prepermute")
        self.permute = module.get_function("permute")
        self.copy_inplace = module.get_function("copy_inplace")
