from typing import Any
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit # type: ignore

def preprocess_module(module: str, args: dict[str, Any], no_extern_c: bool=False):
    '''Replaces special markers <<...>> in cuda source code with
       values provided in the args dictionary.

    '''
    options = [f'-D{key}={value}' for key, value in args.items()]
    return SourceModule(module, no_extern_c=no_extern_c, options=options)


def load_cuda_modules(**args):
    with open("cuda/predict.cu", "r") as f:
        predict = f.read()

    with open("cuda/update.cu", "r") as f:
        update = f.read()

    with open("cuda/rescale.cu", "r") as f:
        rescale = f.read()

    with open("cuda/resample.cu", "r") as f:
        resample = f.read()

    with open("cuda/weights_and_mean_position.cu", "r") as f:
        weights_and_mean = f.read()

    with open("cuda/permute.cu", "r") as f:
        permute = f.read()

    return {
        "predict": preprocess_module(predict, args, no_extern_c=True),
        "update": preprocess_module(update, args),
        "rescale": preprocess_module(rescale, args),
        "resample": preprocess_module(resample, args),
        "weights_and_mean": preprocess_module(weights_and_mean, args),
        "permute": preprocess_module(permute, args)
    }
