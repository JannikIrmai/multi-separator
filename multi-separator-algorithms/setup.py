from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("multi_separator",
                      ["src/multi_separator.cxx"],
                      include_dirs=["include"],
                      define_macros=[('VERSION_INFO', __version__)]),
    Pybind11Extension("multi_cut",
                      ["src/multi_cut.cxx"],
                      include_dirs=["include"],
                      define_macros=[('VERSION_INFO', __version__)]),
    Pybind11Extension("synth_growth",
                      ["src/synth_growth.cxx"],
                      include_dirs=["include"],
                      define_macros=[('VERSION_INFO', __version__)])
]

setup(
    name="multi_separator",
    version=__version__,
    author="Jannik Irmai",
    author_email="jannik.irmai@tu-dresden.de",
    url="https://mlcv.inf.tu-dresden.de/index.html",
    description="Fast heuristics for the multi-separator problem",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
