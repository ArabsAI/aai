import os

import jax

import aai

# ZIPFILE_PATH_GCS = "gs://aai/copies"
ZIPFILE_PATH_GCS = "/tmp/aai/copies"

aai_ROOT_DIR = os.path.dirname(aai.__path__[0])
aai_PATH = aai.__path__[0]


## Jax
JAX_DEFAULT_BACKEND = jax.default_backend()
JAX_DEFAULT_PLATFORM = jax.lib.xla_bridge.get_backend().platform
