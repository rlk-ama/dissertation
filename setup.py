from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("distributions.distributions2",
              sources=["distributions/distributions2.pyx"],
              libraries=["m"] # Unix-like specific
    ),
    Extension("utils.utilsc",
          sources=["utils/utilsc.pyx"],
          libraries=["m"] # Unix-like specific
    ),
        Extension("bootstrap.utils_ricker",
          sources=["bootstrap/utils_ricker.pyx"],
          libraries=["m"] # Unix-like specific
    ),
        Extension("bootstrap.utils_kalman",
          sources=["bootstrap/utils_kalman.pyx"],
          libraries=["m"] # Unix-like specific
    )
]

setup(
  name = "cython",
  ext_modules = cythonize(ext_modules)
)
