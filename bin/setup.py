#!/usr/bin/env python3

"""
*-*-*-*-*-*-*-*-*-  GALAHAD PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-

  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
  Principal author: Jaroslav Fowkes & Nick Gould

  History -
   originally released GALAHAD Version 3.3. August 12th 2021

  For full documentation, see
   http://galahad.rl.ac.uk/galahad-www/specs.html
"""

import numpy as np
import os
from setuptools import setup, Extension, find_packages

GALAHAD = os.getenv('GALAHAD')
print(f'{GALAHAD}')

define_macros=[('LINUX', None)]
include_dirs=[np.get_include(),f'{GALAHAD}''/include/']
libraries=['galahad_py', 'galahad_c', 'galahad_hsl_c', 'galahad', 'galahad_hsl', 'galahad_spral', 'stdc++', 'hwloc', 'galahad_mkl_pardiso', 'galahad_pardiso', 'galahad_wsmp', 'galahad_metis', 'galahad_lapack', 'galahad_blas', 'gfortran', 'galahad_python_dummy']
library_dirs=['/home/nimg/fortran/optrove/galahad/objects/pc64.lnx.gfo/double/dynamic']
extra_link_args=['-Wl,-rpath=/home/nimg/fortran/optrove/galahad/objects/pc64.lnx.gfo/double/dynamic','-lgomp']

# Modules for packages
ugo_module = Extension(
    str('galahad.ugo'),
    sources=[f'{GALAHAD}/src/ugo/python/ugo_pyiface.c'],
#    sources=[],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
bgo_module = Extension(
    str('galahad.bgo'),
    # bgo needs some ugo functions see the following and linked articles:
    # https://stackoverflow.com/questions/57609741/link-against-another-python-c-extension
     sources=[f'{GALAHAD}/src/ugo/python/ugo_pyiface.c',f'{GALAHAD}/src/bgo/python/bgo_pyiface.c'],
#    sources=[],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)

# Main settings
setup(
    name='galahad',
    version='0.1',
    description='Python interfaces to GALAHAD',
    long_description='Python interfaces to the GALAHAD library for nonlinear optimization',
    author='Nick Gould and Jaroslav Fowkes',
    author_email='nick.gould@stfc.ac.uk, jaroslav.fowkes@stfc.ac.uk',
    url='https://github.com/ralna/GALAHAD',
    download_url="https://github.com/ralna/GALAHAD/releases/",
    project_urls={
        "Bug Tracker": "https://github.com/ralna/GALAHAD/issues/",
        "Documentation": "https://www.galahad.rl.ac.uk",
        "Source Code": "https://github.com/ralna/GALAHAD/",
    },
    platforms='Linux',
    license='GNU LGPL',
    packages=find_packages(),
    ext_modules=[ugo_module,bgo_module],
    keywords = "mathematics optimization",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License (LGPL)',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],
)


