from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs

ext_modules = [Extension("pxutil", ["pxutil.pyx"], include_dirs=get_numpy_include_dirs()),
	Extension("procrustesopt", ["procrustesopt.pyx"], include_dirs=get_numpy_include_dirs()),
	Extension("normalisedImageOpt", ["normalisedImageOpt.pyx"], include_dirs=get_numpy_include_dirs()),
	Extension("simpleGbrt", ["simpleGbrt.pyx"], include_dirs=get_numpy_include_dirs()),
	Extension("supraFeatures", ["supraFeatures.pyx"], include_dirs=get_numpy_include_dirs()),
	Extension("lazyhog", ["lazyhog.pyx"], include_dirs=get_numpy_include_dirs())]

setup(
  name = 'supra',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

#python setup.py build_ext --inplace
