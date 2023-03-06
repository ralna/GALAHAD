#!/usr/bin/env python3

"""
*-*-*-*-*-*-*-*-*-  GALAHAD PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-

  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
  Principal author: Jaroslav Fowkes & Nick Gould

  History -
   originally released GALAHAD Version 4.1. August 12th 2022

   This requires that two environment variables have been set

    GALAHAD = main GALAHAD directory
    ARCH = architecture of the form machine.os.compiler indicating the
           machine/operating system/compiler combination to be used
          (see filenames in $GALAHAD/makefiles)

  For full documentation, see
   http://galahad.rl.ac.uk/galahad-www/specs.html
"""

import numpy as np
import os
from setuptools import setup, Extension, find_packages

GALAHAD = os.getenv('GALAHAD')
ARCH = os.getenv('ARCH')
GALAHAD_OBJ = f'{GALAHAD}''/objects/'f'{ARCH}''/double'
GALAHAD_DOBJ = f'{GALAHAD_OBJ}''/shared'

define_macros=[('LINUX', None)]
include_dirs=[np.get_include(),f'{GALAHAD}''/include/', \
              f'{GALAHAD_OBJ}']
libraries=['galahad_py', 'galahad_c', 'galahad_hsl_c', 'galahad', \
           'galahad_hsl', 'galahad_spral', 'stdc++', 'hwloc', \
           'galahad_mkl_pardiso', 'galahad_pardiso', 'galahad_wsmp', \
           'galahad_pastix', 'galahad_mpi', 'galahad_mumps', \
           'galahad_umfpack', \
           'galahad_metis4', 'galahad_lapack', 'galahad_blas', \
           'gfortran', 'galahad_cutest_dummy']
library_dirs=[f'{GALAHAD_DOBJ}']
extra_link_args=['-Wl,-rpath='f'{GALAHAD_DOBJ}','-lgomp']

# Modules for packages
ugo_module = Extension(
    str('galahad.ugo'),
    sources=[f'{GALAHAD}/src/ugo/python/ugo_pyiface.c'],
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
    sources=[f'{GALAHAD}/src/ugo/python/ugo_pyiface.c',\
             f'{GALAHAD}/src/bgo/python/bgo_pyiface.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
dgo_module = Extension(
    str('galahad.dgo'),
    # dgo needs some ugo functions see the following and linked articles:
    # https://stackoverflow.com/questions/57609741/link-against-another-python-c-extension
     sources=[f'{GALAHAD}/src/ugo/python/ugo_pyiface.c',\
              f'{GALAHAD}/src/dgo/python/dgo_pyiface.c'],
#    sources=[],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
arc_module = Extension(
    str('galahad.arc'),
    sources=[f'{GALAHAD}/src/arc/python/arc_pyiface.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
tru_module = Extension(
    str('galahad.tru'),
    sources=[f'{GALAHAD}/src/trb/python/tru_pyiface.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
trb_module = Extension(
    str('galahad.trb'),
    sources=[f'{GALAHAD}/src/tru/python/trb_pyiface.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=define_macros,
    extra_link_args=extra_link_args,
)
nls_module = Extension(
    str('galahad.nls'),
    sources=[f'{GALAHAD}/src/nls/python/nls_pyiface.c'],
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
    ext_modules=[arc_module,tru_module,trb_module,\
                 ugo_module,bgo_module,dgo_module,\
                 nls_module],
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
