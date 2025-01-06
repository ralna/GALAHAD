project(
  'GALAHAD',
  'c', 'cpp', 'fortran',
  version: '5.1.0',
  license: 'BSD-3',
  meson_version: '>= 0.63.0',
  default_options: [
    'buildtype=release',
    'libdir=lib',
    'default_library=shared',
    'warning_level=0',
    'c_std=c99',
    'cpp_std=c++11',
  ],
)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
fc = meson.get_compiler('fortran')
fc_compiler = find_program(fc.cmd_array())
fs = import('fs')
py = import('python').find_installation()
host_system = host_machine.system()

# Remove messages about deprecated Intel compilers
if cc.get_id() == 'intel'
  add_global_arguments('-diag-disable=10441', language : 'c')
  add_global_link_arguments('-diag-disable=10441', language : 'c')
endif
if cc.get_id() == 'intel-cl'
  add_global_arguments('/Qdiag-disable=10441', language : 'c')
  add_global_link_arguments('/Qdiag-disable=10441', language : 'c')
endif
if cxx.get_id() == 'intel'
  add_global_arguments('-diag-disable=10441', language : 'cpp')
  add_global_link_arguments('-diag-disable=10441', language : 'cpp')
endif
if cxx.get_id() == 'intel-cl'
  add_global_arguments('/Qdiag-disable=10441', language : 'cpp')
  add_global_link_arguments('/Qdiag-disable=10441', language : 'cpp')
endif

# Recognise old non-standard double complex intrinsics
if fc.get_id() == 'nagfor'
  add_global_arguments('-dcfuns', language : 'fortran')
endif

# Preprocessing
if fc.get_id() == 'intel-cl' or fc.get_id() == 'intel-llvm-cl'
  output_generator = '/Fi:@OUTPUT@'
else
  output_generator = ['-o', '@OUTPUT@']
endif

pp_options = []
pp_flag = []
if fc.get_id() == 'gcc'
  pp_flag += '-cpp'
  pp_options += ['-cpp', '-E']
elif fc.get_id() == 'intel' or fc.get_id() == 'intel-llvm'
  pp_flag += '-fpp'
  pp_options += ['-fpp', '-P']
elif fc.get_id() == 'intel-cl' or fc.get_id() == 'intel-llvm-cl'
  pp_flag += '/fpp'
  pp_options += ['/fpp', '/P']
elif fc.get_id() == 'nagfor'
  pp_flag += '-fpp'
  pp_options += ['-fpp', '-F']
elif fc.get_id() == 'nvidia_hpc'
  pp_flag += '-Mcpp'
  pp_options += ['-Mcpp', '-F']
endif

# Options
install_modules = get_option('modules')
int64 = get_option('int64')
multiprecision = get_option('multiprecision')
build_ciface = get_option('ciface')
build_pythoniface = get_option('pythoniface')
build_single = get_option('single')
build_double = get_option('double')
build_quadruple = get_option('quadruple')
build_tests = get_option('tests')
build_examples = get_option('examples')
build_binaries = get_option('binaries')
build_ssids = get_option('ssids')

libblas_name = get_option('libblas')
liblapack_name = get_option('liblapack')
libhsl_name = get_option('libhsl')
libwsmp_name = get_option('libwsmp')
libpardiso_name = get_option('libpardiso')
libspmf_name = get_option('libspmf')
libpastixf_name = get_option('libpastixf')
libsmumps_name = get_option('libsmumps')
libdmumps_name = get_option('libdmumps')
libampl_name = get_option('libampl')
libcutest_single_name = get_option('libcutest_single')
libcutest_double_name = get_option('libcutest_double')
libcutest_quadruple_name = get_option('libcutest_quadruple')
libhwloc_name = get_option('libhwloc')

libmumps_path = get_option('libmumps_path')
libblas_path = get_option('libblas_path')
liblapack_path = get_option('liblapack_path')
libhsl_path = get_option('libhsl_path')
libwsmp_path = get_option('libwsmp_path')
libpardiso_path = get_option('libpardiso_path')
libspmf_path = get_option('libspmf_path')
libpastixf_path = get_option('libpastixf_path')
libampl_path = get_option('libampl_path')
libcutest_path = get_option('libcutest_path')
libhwloc_path = get_option('libhwloc_path')

libhwloc_include = include_directories(get_option('libhwloc_include'))
libhsl_modules = include_directories(get_option('libhsl_modules'))
libcutest_modules = include_directories(get_option('libcutest_modules'))

# Dependencies
libblas = fc.find_library(libblas_name, dirs : libblas_path, required : false)
liblapack = fc.find_library(liblapack_name, dirs : liblapack_path, required : false)
libhsl = fc.find_library(libhsl_name, dirs : libhsl_path, required : false)
libcutest_single = fc.find_library(libcutest_single_name, dirs : libcutest_path, required : false)
libcutest_double = fc.find_library(libcutest_double_name, dirs : libcutest_path, required : false)
libcutest_quadruple = fc.find_library(libcutest_quadruple_name, dirs : libcutest_path, required : false)
libwsmp = fc.find_library(libwsmp_name, dirs : libwsmp_path, required : false)
libpardiso = fc.find_library(libpardiso_name, dirs : libpardiso_path, required : false)
libspmf = fc.find_library(libspmf_name, dirs : libspmf_path, required : false)
libpastixf = fc.find_library(libpastixf_name, dirs : libpastixf_path, required : false)
libsmumps = fc.find_library(libsmumps_name, dirs : libmumps_path, required : false)
libdmumps = fc.find_library(libdmumps_name, dirs : libmumps_path, required : false)
libampl = fc.find_library(libampl_name, dirs : libampl_path, required : false)
libhwloc = fc.find_library(libhwloc_name, dirs : libhwloc_path, required : false)
lm = cc.find_library('m', required : false)
has_hwloch = cc.has_header('hwloc.h', include_directories : libhwloc_include)

# Quadmath
if build_quadruple
  libquadmath = cc.find_library('quadmath', required : true)
else
  libquadmath = cc.find_library('quadmath', required : false)
endif

# METIS
if host_machine.system() == 'windows'
  add_global_arguments('-DUSE_GKREGEX', language : 'c')
endif

# OpenMP
if fc.get_id() == 'nvidia_hpc'
  add_global_arguments('-mp', language : 'fortran')
  add_global_link_arguments('-mp', language : 'fortran')
elif fc.get_id() == 'nagfor'
  add_global_arguments('-openmp', language : 'fortran')
  add_global_link_arguments('-openmp', language : 'fortran')
elif fc.get_id() == 'gcc'
  add_global_arguments('-fopenmp', language : 'fortran')
  add_global_link_arguments('-fopenmp', language : 'fortran')
elif fc.get_id() == 'intel' or fc.get_id() == 'intel-llvm'
  add_global_arguments('-qopenmp', language : 'fortran')
  add_global_link_arguments('-qopenmp', language : 'fortran')
elif fc.get_id() == 'intel-cl' or fc.get_id() == 'intel-llvm-cl'
  add_global_arguments('/Qopenmp', language : 'fortran')
endif

if cc.get_id() == 'nvidia_hpc'
  add_global_arguments('-mp', language : 'c')
  add_global_link_arguments('-mp', language : 'c')
elif cc.get_id() == 'gcc' or cc.get_id() == 'clang' or cc.get_id() == 'clang-cl'
  add_global_arguments('-fopenmp', language : 'c')
  add_global_link_arguments('-fopenmp', language : 'c')
elif cc.get_id() == 'intel' or cc.get_id() == 'intel-llvm'
  add_global_arguments('-qopenmp', language : 'c')
  add_global_link_arguments('-qopenmp', language : 'c')
elif cc.get_id() == 'intel-cl' or cc.get_id() == 'intel-llvm-cl'
  add_global_arguments('/Qopenmp', language : 'c')
endif

if cxx.get_id() == 'nvidia_hpc'
  add_global_arguments('-mp', language : 'cpp')
elif cxx.get_id() == 'gcc' or cxx.get_id() == 'clang' or cxx.get_id() == 'clang-cl'
  add_global_arguments('-fopenmp', language : 'cpp')
elif cxx.get_id() == 'intel' or cxx.get_id() == 'intel-llvm'
  add_global_arguments('-qopenmp', language : 'cpp')
elif cxx.get_id() == 'intel-cl' or cxx.get_id() == 'intel-llvm-cl'
  add_global_arguments('/Qopenmp', language : 'cpp')
endif

libgalahad_single_deps = [libsmumps, libcutest_single, libblas, liblapack,
                          libhsl, libwsmp, libpardiso, libspmf, libpastixf,
                          libampl, libhwloc]

libgalahad_double_deps = [libdmumps, libcutest_double, libblas, liblapack,
                          libhsl, libwsmp, libpardiso, libspmf, libpastixf,
                          libampl, libhwloc]

libgalahad_quadruple_deps = [libcutest_quadruple, libhsl, libwsmp, libpardiso, libspmf,
                             libpastixf, libampl, libhwloc]

# BLAS and LAPACK
libblas_src = []
liblapack_src = []

# Fortran files *.F90 or *.f90
libgalahad_src = []
libgalahad_single_src = []
libgalahad_double_src = []
libgalahad_quadruple_src = []

# Fortran files *.f
libgalahad_f_src = []

# Fortran files that contain programs
galahad_binaries = []

# Header files to install
galahad_headers = []

# Fortran files for the C interfaces
libgalahad_c_src = []
libgalahad_c_single_src = []
libgalahad_c_double_src = []
libgalahad_c_quadruple_src = []

# C files for the Python interfaces
libgalahad_python_src = []

# Files related to the CUTEst interfaces
libgalahad_cutest_src = []
galahad_cutest_binaries = []

# C and C++ files required by GALAHAD
libgalahad_cpp_src = []
libgalahad_cc_src = []

# Fortran examples and tests
galahad_examples = []
galahad_tests = []

# C examples and tests for the C interfaces
galahad_c_examples = []
galahad_c_tests = []

# Python tests for the Python interfaces
galahad_python_tests = []

# Folders that contain headers and Fortran modules
libgalahad_include = [include_directories('include'),
                      include_directories('src/dum/include'),
                      include_directories('src/metis/include'),
                      include_directories('src/ampl')] + libhsl_modules + libcutest_modules + libhwloc_include

if host_machine.system() == 'linux'
  add_global_arguments('-DSPRAL_HAVE_SCHED_GETCPU', language : 'cpp')
else
  add_global_arguments('-DSPRAL_NO_SCHED_GETCPU', language : 'cpp')
endif

# HWLOC
if libhwloc.found() and has_hwloch
  add_global_arguments('-DSPRAL_HAVE_HWLOC', language : 'cpp')
else
  add_global_arguments('-DSPRAL_NO_HWLOC', language : 'cpp')
endif

# HSL
if libhsl.found()
  add_global_arguments('-DLANCELOT_USE_MA57', language : 'fortran')
endif

# Link flag for C++
lstdcpp = '-lstdc++'
if host_machine.system() != 'linux'
  if cxx.get_id() == 'clang' or cxx.get_id() == 'clang-cl'
    lstdcpp = '-lc++'
  endif
  if cxx.get_id() == 'intel' or cxx.get_id() == 'intel-llvm'
    lstdcpp = '-cxxlib'
  endif
  if cxx.get_id() == 'intel-cl' or cxx.get_id() == 'intel-llvm-cl'
    lstdcpp = '/Qcxxlib'
  endif
endif

extra_args_single = ['-DREAL_32']
extra_args_double = []
extra_args_quadruple = ['-DREAL_128', '-DGALAHAD_BLAS', '-DGALAHAD_LAPACK', '-DDUMMY_QMUMPS']

if not libblas.found()
  extra_args_single += '-DGALAHAD_BLAS'
  extra_args_double += '-DGALAHAD_BLAS'
endif
if not liblapack.found()
  extra_args_single += '-DGALAHAD_LAPACK'
  extra_args_double += '-DGALAHAD_LAPACK'
endif
if not libsmumps.found()
  extra_args_single += '-DDUMMY_SMUMPS'
endif
if not libdmumps.found()
  extra_args_double += '-DDUMMY_DMUMPS'
endif
if not (libblas_name == 'mkl_rt' or liblapack_name == 'mkl_rt')
  extra_args_single += '-DDUMMY_MKL_PARDISO'
  extra_args_double += '-DDUMMY_MKL_PARDISO'
  extra_args_quadruple += '-DDUMMY_MKL_PARDISO'
endif
if not libpardiso.found()
  extra_args_single += '-DDUMMY_PARDISO'
  extra_args_double += '-DDUMMY_PARDISO'
  extra_args_quadruple += '-DDUMMY_PARDISO'
endif
if not libpastixf.found()
  extra_args_single += '-DDUMMY_PASTIXF'
  extra_args_double += '-DDUMMY_PASTIXF'
  extra_args_quadruple += '-DDUMMY_PASTIXF'
endif
if not libspmf.found()
  extra_args_single += '-DDUMMY_SPMF'
  extra_args_double += '-DDUMMY_SPMF'
  extra_args_quadruple += '-DDUMMY_SPMF'
endif
if not libwsmp.found()
  extra_args_single += '-DDUMMY_WSMP'
  extra_args_double += '-DDUMMY_WSMP'
  extra_args_quadruple += '-DDUMMY_WSMP'
endif
if not libhsl.found()
  extra_args_single += '-DDUMMY_HSL'
  extra_args_double += '-DDUMMY_HSL'
  extra_args_quadruple += '-DDUMMY_HSL'
endif

# MPI
extra_args_single += '-DDUMMY_MPI'
extra_args_double += '-DDUMMY_MPI'
extra_args_quadruple += '-DDUMMY_MPI'

if multiprecision
  extra_args_single += '-DMULTIPRECISION'
  extra_args_double += '-DMULTIPRECISION'
  extra_args_quadruple += '-DMULTIPRECISION'
endif

# Compile GALAHAD with 64-bit integer
if int64
  extra_args_single += '-DINTEGER_64'
  extra_args_double += '-DINTEGER_64'
  extra_args_quadruple += '-DINTEGER_64'
endif

# Sources
subdir('bin')
subdir('include')
subdir('src/amd')
subdir('src/ampl')
subdir('src/arc')
subdir('src/band')
subdir('src/bgo')
subdir('src/blls')
subdir('src/bllsb')
# subdir('src/bnls')
subdir('src/bqp')
subdir('src/bqpb')
subdir('src/bsc')
subdir('src/buildspec')
subdir('src/ccqp')
subdir('src/cdqp')
subdir('src/check')
subdir('src/checkpoint')
subdir('src/clls')
subdir('src/clock')
subdir('src/common')
subdir('src/convert')
subdir('src/copyright')
subdir('src/cqp')
subdir('src/cro')
subdir('src/cutest_functions')
subdir('src/demo')
subdir('src/dgo')
subdir('src/dlp')
subdir('src/dps')
subdir('src/dqp')
subdir('src/dum')
subdir('src/eqp')
subdir('src/extend')
subdir('src/external')
subdir('src/fdc')
subdir('src/fdh')
subdir('src/filter')
subdir('src/filtrane')
subdir('src/fit')
subdir('src/glrt')
subdir('src/gls')
subdir('src/gltr')
subdir('src/hash')
subdir('src/icfs')
subdir('src/ir')
subdir('src/kinds')
subdir('src/l1qp')
subdir('src/l2rt')
subdir('src/lancelot')
subdir('src/lancelot_simple')
subdir('src/lapack')
subdir('src/lhs')
subdir('src/lls')
subdir('src/llsr')
subdir('src/llst')
subdir('src/lms')
subdir('src/lmt')
subdir('src/lpa')
subdir('src/lpb')
subdir('src/lpqp')
subdir('src/lqr')
subdir('src/lqt')
subdir('src/lsp')
subdir('src/lsqp')
subdir('src/lsrb')
subdir('src/lsrt')
subdir('src/lstr')
subdir('src/metis')
subdir('src/miqr')
subdir('src/mop')
subdir('src/nlpt')
subdir('src/nls')
subdir('src/norms')
subdir('src/oblivion')
subdir('src/opt')
subdir('src/presolve')
subdir('src/problem')
subdir('src/psls')
subdir('src/ptrans')
subdir('src/qp')
subdir('src/qpa')
subdir('src/qpb')
subdir('src/qpc')
subdir('src/qpd')
subdir('src/qpp')
subdir('src/qpt')
subdir('src/rand')
subdir('src/roots')
subdir('src/rpd')
subdir('src/rqs')
subdir('src/sbls')
subdir('src/scale')
subdir('src/scu')
subdir('src/sec')
subdir('src/sha')
subdir('src/sils')
subdir('src/slls')
subdir('src/sls')
subdir('src/smt')
subdir('src/sort')
subdir('src/space')
subdir('src/specfile')
subdir('src/spral')
subdir('src/ssids')
subdir('src/string')
subdir('src/svt')
subdir('src/symbols')
subdir('src/tools')
subdir('src/trans')
subdir('src/trb')
subdir('src/trs')
subdir('src/tru')
subdir('src/ugo')
subdir('src/uls')
subdir('src/userdata')
subdir('src/version')
subdir('src/warm')
subdir('src/wcp')

# Libraries
if build_single
  gen_single = generator(fc_compiler,
                         output : 'single_@BASENAME@.f90',
                         arguments : pp_options + extra_args_single +
                                     ['-I', '@CURRENT_SOURCE_DIR@/include',
                                      '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                      '@INPUT@'] + output_generator)

  gen_f_single = generator(fc_compiler,
                           output : 'single_@BASENAME@.f',
                           arguments : pp_options + extra_args_single +
                                       ['-I', '@CURRENT_SOURCE_DIR@/include',
                                        '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                        '@INPUT@'] + output_generator)

  to_process_single = [libgalahad_src, libgalahad_single_src]
  if build_ciface
    to_process_single += libgalahad_c_src
    to_process_single += libgalahad_c_single_src
  endif
  if libcutest_single.found()
    to_process_single += libgalahad_cutest_src
  else
    to_process_single += libgalahad_cutest_dummy_src
  endif
  pp_sources_single = gen_single.process(to_process_single)

  to_process_f_single = [libgalahad_f_src]
  if to_process_f_single == [[]]
    pp_sources_f_single = []
  else
    pp_sources_f_single = gen_f_single.process(to_process_f_single)
  endif

  sources_single = [pp_sources_single, pp_sources_f_single, libgalahad_cpp_src, libgalahad_cc_src]

  libgalahad_single_name = 'galahad_single'
  if int64
    libgalahad_single_name += '_64'
  endif

  libgalahad_single = library(libgalahad_single_name,
                              sources : sources_single,
                              dependencies : libgalahad_single_deps,
                              fortran_args : extra_args_single,
                              c_args : extra_args_single,
                              cpp_args : extra_args_single,
                              link_language : 'fortran',
                              link_args : lstdcpp,
                              include_directories: libgalahad_include,
                              install : true)
endif

if build_double
  gen_double = generator(fc_compiler,
                         output : 'double_@BASENAME@.f90',
                         arguments : pp_options + extra_args_double +
                                     ['-I', '@CURRENT_SOURCE_DIR@/include',
                                      '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                      '@INPUT@'] + output_generator)

  gen_f_double = generator(fc_compiler,
                           output : 'double_@BASENAME@.f',
                           arguments : pp_options + extra_args_double +
                                       ['-I', '@CURRENT_SOURCE_DIR@/include',
                                        '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                        '@INPUT@'] + output_generator)

  to_process_double = [libgalahad_src, libgalahad_double_src]
  if build_ciface
    to_process_double += libgalahad_c_src
    to_process_double += libgalahad_c_double_src
  endif
  if libcutest_double.found()
    to_process_double += libgalahad_cutest_src
  else
    to_process_double += libgalahad_cutest_dummy_src
  endif
  pp_sources_double = gen_double.process(to_process_double)

  to_process_f_double = [libgalahad_f_src]
  if to_process_f_double == [[]]
    pp_sources_f_double = []
  else
    pp_sources_f_double = gen_f_double.process(to_process_f_double)
  endif

  sources_double = [pp_sources_double, pp_sources_f_double, libgalahad_cpp_src, libgalahad_cc_src]

  libgalahad_double_name = 'galahad_double'
  if int64
    libgalahad_double_name += '_64'
  endif

  libgalahad_double = library(libgalahad_double_name,
                              sources : sources_double,
                              dependencies : libgalahad_double_deps,
                              fortran_args : extra_args_double,
                              c_args : extra_args_double,
                              cpp_args : extra_args_double,
                              link_language : 'fortran',
                              link_args : lstdcpp,
                              include_directories: libgalahad_include,
                              install : true)
endif

if build_quadruple
  gen_quadruple = generator(fc_compiler,
                            output : 'quadruple_@BASENAME@.f90',
                            arguments : pp_options + extra_args_quadruple +
                                        ['-I', '@CURRENT_SOURCE_DIR@/include',
                                         '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                         '@INPUT@'] + output_generator)

  gen_f_quadruple = generator(fc_compiler,
                              output : 'quadruple_@BASENAME@.f',
                              arguments : pp_options + extra_args_quadruple +
                                          ['-I', '@CURRENT_SOURCE_DIR@/include',
                                           '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                           '@INPUT@'] + output_generator)

  to_process_quadruple = [libgalahad_src, libgalahad_quadruple_src]
  if build_ciface
    to_process_quadruple += libgalahad_c_src
    to_process_quadruple += libgalahad_c_quadruple_src
  endif
  if libcutest_quadruple.found()
    to_process_quadruple += libgalahad_cutest_src
  else
    to_process_quadruple += libgalahad_cutest_dummy_src
  endif
  pp_sources_quadruple = gen_quadruple.process(to_process_quadruple)

  to_process_f_quadruple = [libgalahad_f_src]
  if to_process_f_quadruple == [[]]
    pp_sources_f_quadruple = []
  else
    pp_sources_f_quadruple = gen_f_quadruple.process(to_process_f_quadruple)
  endif

  sources_quadruple = [pp_sources_quadruple, pp_sources_f_quadruple, libgalahad_cpp_src, libgalahad_cc_src]

  libgalahad_quadruple_name = 'galahad_quadruple'
  if int64
    libgalahad_quadruple_name += '_64'
  endif

  libgalahad_quadruple = library(libgalahad_quadruple_name,
                              sources : sources_quadruple,
                              dependencies : libgalahad_quadruple_deps,
                              fortran_args : extra_args_quadruple,
                              c_args : extra_args_quadruple,
                              cpp_args : extra_args_quadruple,
                              link_language : 'fortran',
                              link_args : lstdcpp,
                              include_directories: libgalahad_include,
                              install : true)
endif

# Python interface
if build_pythoniface and build_ciface and build_double and (not int64)
  incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check : true
  ).stdout().strip()

  # install python metadata
  install_data(files('doc/PYTHON_METADATA'), install_dir : join_paths(py.get_install_dir(),'galahad-'+meson.project_version()+'.dist-info'))

  # compile and install python interfaces
  foreach interface: libgalahad_python_src
    name = interface[0]
    file = interface[1]

    py.extension_module(name,
                        sources : file,
                        link_with : libgalahad_double,
                        link_language : 'c',
                        include_directories : libgalahad_include + incdir_numpy,
                        subdir : 'galahad',
                        install : true)
  endforeach
endif

# Binaries
if build_binaries
  galahad_binaries_single = galahad_binaries
  galahad_binaries_double = galahad_binaries
  galahad_binaries_quadruple = galahad_binaries
  if libcutest_single.found()
    galahad_binaries_single += galahad_cutest_binaries
  endif
  if libcutest_double.found()
    galahad_binaries_double += galahad_cutest_binaries
  endif
  if libcutest_quadruple.found()
    galahad_binaries_quadruple += galahad_cutest_binaries
  endif

  if build_single
    foreach binary: galahad_binaries_single
      binname = binary[0]
      binfile = binary[1]
      executable(binname+'_single', binfile,
                 dependencies : libgalahad_single_deps,
                 fortran_args : pp_flag + extra_args_single,
                 link_with : libgalahad_single,
                 link_language : 'fortran',
                 include_directories: libgalahad_include,
                 install : true)
    endforeach
  endif

  if build_double
    foreach binary: galahad_binaries_double
      binname = binary[0]
      binfile = binary[1]
      executable(binname+'_double', binfile,
                 dependencies : libgalahad_double_deps,
                 fortran_args : pp_flag + extra_args_double,
                 link_with : libgalahad_double,
                 link_language : 'fortran',
                 include_directories: libgalahad_include,
                 install : true)
    endforeach
  endif

  if build_quadruple
    foreach binary: galahad_binaries_quadruple
      binname = binary[0]
      binfile = binary[1]
      executable(binname+'_quadruple', binfile,
                 dependencies : libgalahad_quadruple_deps,
                 fortran_args : pp_flag + extra_args_quadruple,
                 link_with : libgalahad_quadruple,
                 link_language : 'fortran',
                 include_directories: libgalahad_include,
                 install : true)
    endforeach
  endif
endif

# Headers
install_headers(galahad_headers)

# Fortran modules
if install_modules
  meson.add_install_script(script_modules)
endif

# Precisions
precisions = []
if build_single
  precisions += [['single', extra_args_single, libgalahad_single]]
endif
if build_double
  precisions += [['double', extra_args_double, libgalahad_double]]
endif
if build_quadruple
  precisions += [['quadruple', extra_args_quadruple, libgalahad_quadruple]]
endif

# Fortran tests
if build_tests

  fortran_tests_folder = 'tests/Fortran'

  foreach test: galahad_tests
    foreach val: precisions
      precision = val[0]
      args_precision = val[1]
      libgalahad_precision = val[2]
      package = test[0]
      name = test[1] + '_' + precision
      file = test[2]

      if precision == 'single'
        deps_tests = libgalahad_single_deps
      endif
      if precision == 'double'
        deps_tests = libgalahad_double_deps
      endif
      if precision == 'quadruple'
        deps_tests = libgalahad_quadruple_deps
      endif

      if not (name == 'croti_single' or name == 'croti_quadruple')
        test(name,
             executable(name, file,
                        fortran_args : pp_flag + args_precision,
                        link_with : libgalahad_precision,
                        dependencies : deps_tests,
                        link_language : 'fortran',
                        include_directories: libgalahad_include,
                        install : true,
                        install_dir : fortran_tests_folder),
             suite : [package, precision, 'fortran'],
             is_parallel : false)
      endif
    endforeach
  endforeach
endif

# C tests
if build_tests and build_ciface

  c_tests_folder = 'tests/C'

  foreach test: galahad_c_tests
    foreach val: precisions
      precision = val[0]
      args_precision = val[1]
      libgalahad_precision = val[2]
      package = test[0]
      name = test[1] + '_' + precision
      file = test[2]
      test(name,
           executable(name, file,
                      c_args : args_precision, link_with : libgalahad_precision,
                      dependencies : [lm, libquadmath], link_language : 'c',
                      include_directories : libgalahad_include, install : true,
                      install_dir : c_tests_folder),
           suite : [package, precision, 'C'],
           is_parallel : false)
    endforeach
  endforeach
endif

# Python tests
if build_tests and build_pythoniface and build_ciface and build_double and (not int64) and (host_system != 'windows')

  foreach test: galahad_python_tests
    package = test[0]
    name = test[1]
    file = test[2]
    test(name,
         py,
         args : file,
         suite : [package, 'Python'],
         env : ['PYTHONPATH=' + py.get_install_dir()],
         is_parallel : false)
  endforeach
endif

# Fortran examples
if build_examples and build_double and (not int64)

  fortran_examples_folder = 'examples/Fortran'

  foreach example: galahad_examples
    name = example[0]
    file = example[1]
    executable(name, file, link_with : libgalahad_double,
               dependencies : libgalahad_double_deps, link_language : 'fortran',
               include_directories : libgalahad_include,
               install : true, install_dir : fortran_examples_folder)
  endforeach
endif

# C examples
if build_examples and build_ciface

  c_examples_folder = 'examples/C'

  foreach example: galahad_c_examples
    foreach val: precisions
      precision = val[0]
      args_precision = val[1]
      libgalahad_precision = val[2]
      name = example[0] + '_' + precision
      file = example[1]
      executable(name, file, c_args : args_precision, link_with : libgalahad_precision,
                 dependencies : [lm, libquadmath], link_language : 'c',
                 include_directories : libgalahad_include,
                 install : true, install_dir : c_examples_folder)
    endforeach
  endforeach
endif
