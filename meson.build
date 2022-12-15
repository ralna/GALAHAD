project(
  'GALAHAD',
  'fortran',
  'c',
  version: '5.0.0',
  license: 'BSD-3',
  meson_version: '>= 0.61.0',
  default_options: [
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

fc = meson.get_compiler('fortran')
fs = import('fs')  # filesystem module

extra_args = []
if fc.get_id() == 'nagfor'
  extra_args += '-fixed'
  extra_args += '-dcfuns'
endif

libgalahad_link_with = []
libgalahad_include = ['include']
libgalahad_src = []
libgalahad_c_src = []
libgalahad_blas_src = []
libgalahad_lapack_src = []
subdir('src/lapack')  # libgalahad_blas/lapack only contain the f90 interfaces
libgalahad_deps = []
blas_dep = []
lapack_dep = []

extra_args_single = ['-DGALAHAD_SINGLE']
extra_args_double = ['-DGALAHAD_DOUBLE']

# libgalahad_blas

libblas_opt = get_option('libblas')
if libblas_opt == [] or libblas_opt.length() != 2
  warning('building our own BLAS; consider providing an optimized BLAS library')
  libgalahad_blas = library('galahad_blas',
                            sources : libgalahad_blas_src,
                            fortran_args : extra_args,
                            install : true)
  libgalahad_link_with += libgalahad_blas
else
  libblas = fc.find_library(libblas_opt[1], dirs : libblas_opt[0], required : true)
  blas_dep += libblas
endif

# libgalahad_lapack

liblapack_opt = get_option('liblapack')
if liblapack_opt == [] or liblapack_opt.length() != 2
  warning('building our own LAPACK; consider providing an optimized LAPACK library')
  libgalahad_lapack = library('galahad_lapack',
                              sources : libgalahad_lapack_src,
                              link_with : libgalahad_blas,
                              dependencies : blas_dep,
                              fortran_args : extra_args,
                              install : true)
  libgalahad_link_with += libgalahad_lapack
else
  liblapack = fc.find_library(liblapack_opt[1], dirs : liblapack_opt[0], required : true)
  lapack_dep += liblapack
endif

libgalahad_deps += blas_dep + lapack_dep

# Each subdir() command includes a meson.build in a subdirectory
# that adds to the list of source files.

# libgalahad
subdir('src/precision')
subdir('src/common')
subdir('src/symbols')
subdir('src/specfile')
subdir('src/zd11')
subdir('src/smt')
subdir('src/clock')
subdir('src/space')
subdir('src/userdata')
subdir('src/ugo')

libgalahad_single = library('galahad_single',
                    sources : libgalahad_src,
                    link_with : libgalahad_link_with,
                    dependencies : libgalahad_deps,
                    fortran_args : extra_args_single,
                    include_directories: libgalahad_include,
                    install : true)

libgalahad_double = library('galahad_double',
                    sources : libgalahad_src,
                    link_with : libgalahad_link_with,
                    dependencies : libgalahad_deps,
                    fortran_args : extra_args_double,
                    include_directories: libgalahad_include,
                    install : true)

build_ciface = get_option('ciface')

if build_ciface
libgalahad_c_single = library('galahad_c_single',
                      sources : libgalahad_c_src,
                      link_with : libgalahad_single,
                      fortran_args : extra_args_single,
                      include_directories: libgalahad_include,
                      install : true)

  libgalahad_c_double = library('galahad_c_double',
                        sources : libgalahad_c_src,
                        link_with : libgalahad_double,
                        fortran_args : extra_args_double,
                        include_directories: libgalahad_include,
                        install : true)
endif

# Example programs

share_examples = 'share/examples'
# executable('ugos_single', 'src/ugo/ugos.f90', fortran_args : extra_args_single, link_with : libgalahad_single, install : true, install_dir : share_examples, include_directories: libgalahad_include)
executable('ugos_double', 'src/ugo/ugos.f90', fortran_args : extra_args_double, link_with : libgalahad_double, install : true, install_dir : share_examples, include_directories: libgalahad_include)

# Unit tests

ugot_single = executable('ugot_single', 'src/ugo/ugot.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ugot_double = executable('ugot_double', 'src/ugo/ugot.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('ugo_single', ugot_single, suite: 'single', is_parallel : false)
test('ugo_double', ugot_double, suite: 'double', is_parallel : false)

if build_ciface
  ugot_c_single = executable('ugot_c_single', 'src/ugo/C/ugot.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  ugot_c_double = executable('ugot_c_double', 'src/ugo/C/ugot.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('ugo_c_single', ugot_c_single, suite: ['C', 'single'], is_parallel : false)
  test('ugo_c_double', ugot_c_double, suite: ['C', 'double'], is_parallel : false)
endif
