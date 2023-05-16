project(
  'GALAHAD',
  'c', 'cpp', 'fortran',
  version: '5.0.0',
  license: 'BSD-3',
  meson_version: '>= 0.63.0',
  default_options: [
    'buildtype=debug',
    'libdir=lib',
    'default_library=both',
    'warning_level=0',
    'c_std=c99',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
fc = meson.get_compiler('fortran')
fc_compiler = find_program(fc.cmd_array())
fs = import('fs')
py = import('python').find_installation()

# Remove messages about deprecated Intel compilers
if cc.get_id() == 'intel' or cc.get_id() == 'intel-cl'
  add_global_arguments('-diag-disable=10441', language : 'c')
  add_global_link_arguments('-diag-disable=10441', language : 'c')
endif
if cxx.get_id() == 'intel' or cxx.get_id() == 'intel-cl'
  add_global_arguments('-diag-disable=10441', language : 'cpp')
  add_global_link_arguments('-diag-disable=10441', language : 'cpp')
endif

# Recognise old non-standard double complex intrinsics
if fc.get_id() == 'nagfor'
  add_global_arguments('-dcfuns', language : 'fortran')
endif

# Preprocessing
pp_options = []
if fc.get_id() == 'gcc'
  pp_options += ['-cpp', '-E']
elif fc.get_id() == 'intel' or fc.get_id() == 'intel-cl'
  pp_options += ['-fpp', '-P']
elif fc.get_id() == 'nagfor'
  pp_options += ['-fpp', '-F']
endif

if cxx.get_id() == 'gcc' or cxx.get_id() == 'intel' or cxx.get_id() == 'intel-cl'
  add_global_arguments('-MD', language : 'cpp')
endif

# Options
install_modules = get_option('modules')

build_ciface = get_option('ciface')
build_pythoniface = get_option('pythoniface')
build_single = get_option('single')
build_double = get_option('double')
build_tests = get_option('tests')
build_examples = get_option('examples')
build_ssids = get_option('ssids')

libblas_name = get_option('libblas')
liblapack_name = get_option('liblapack')
libmetis_name = get_option('libmetis')
libhsl_name = get_option('libhsl')
libcutest_name = get_option('libcutest')
libwsmp_name = get_option('libwsmp')
libumfpack_name = get_option('libumfpack')
libpardiso_name = get_option('libpardiso')
libspmf_name = get_option('libspmf')
libpastix_name = get_option('libpastix')
libmkl_pardiso_name = get_option('libmkl_pardiso')
libampl_name = get_option('libampl')

libmumps_path = get_option('libmumps_path')
libblas_path = get_option('libblas_path')
liblapack_path = get_option('liblapack_path')
libmetis_path = get_option('libmetis_path')
libhsl_path = get_option('libhsl_path')
libcutest_path = get_option('libcutest_path')
libwsmp_path = get_option('libwsmp_path')
libumfpack_path = get_option('libumfpack_path')
libpardiso_path = get_option('libpardiso_path')
libspmf_path = get_option('libspmf_path')
libpastix_path = get_option('libpastix_path')
libmkl_pardiso_path = get_option('libmkl_pardiso_path')
libampl_path = get_option('libampl_path')

libmetis_version = get_option('libmetis_version')

libhsl_modules = get_option('libhsl_modules')

# Dependencies
libblas = fc.find_library(libblas_name, dirs : libblas_path, required : false)
liblapack = fc.find_library(liblapack_name, dirs : liblapack_path, required : false)
libmetis = fc.find_library(libmetis_name, dirs : libmetis_path, required : false)
libhsl = fc.find_library(libhsl_name, dirs : libhsl_path, required : false)
libcutest = fc.find_library(libcutest_name, dirs : libcutest_path, required : false)
libwsmp = fc.find_library(libwsmp_name, dirs : libwsmp_path, required : false)
libumfpack = fc.find_library(libumfpack_name, dirs : libumfpack_path, required : false)
libpardiso = fc.find_library(libpardiso_name, dirs : libpardiso_path, required : false)
libspmf = fc.find_library(libspmf_name, dirs : libspmf_path, required : false)
libpastix = fc.find_library(libpastix_name, dirs : libpastix_path, required : false)
libsmumps = fc.find_library('smumps', dirs : libmumps_path, required : false)
libdmumps = fc.find_library('dmumps', dirs : libmumps_path, required : false)
libmkl_pardiso = fc.find_library(libmkl_pardiso_name, dirs : libmkl_pardiso_path, required : false)
libampl = fc.find_library(libampl_name, dirs : libampl_path, required : false)
lm = cc.find_library('m', required : false)

# OpenMP
if fc.get_id() == 'nagfor'
  add_global_link_arguments('-openmp', language : 'fortran')
elif fc.get_id() == 'gcc'
  add_global_link_arguments('-fopenmp', language : 'fortran')
elif fc.get_id() == 'intel' or fc.get_id() == 'intel-cl'
  add_global_link_arguments('-qopenmp', language : 'fortran')
endif

if cxx.get_id() == 'gcc'
  add_global_arguments('-fopenmp', language : 'cpp')
endif
# --- FIX ME ---
# if cxx.get_id() == 'intel' or cxx.get_id() == 'intel-cl'
#   add_global_arguments('-qopenmp', language : 'cpp')
# endif

libmpi = dependency('mpi', language : 'fortran', required : false)
libhwloc = dependency('hwloc', required : false)

libgalahad_single_deps = [libsmumps]
libgalahad_double_deps = [libdmumps]
libgalahad_deps = [libblas, liblapack, libmetis, libhsl, libcutest,
                   libwsmp, libumfpack, libpardiso, libspmf, libpastix,
                   libmkl_pardiso, libampl, libmpi, libhwloc]

libgalahad_single_src = []
libgalahad_double_src = []
libgalahad_integer_src = []
libgalahad_multi_src = []
libgalahad_src = []
libgalahad_aux = []

galahad_headers = []
galahad_binaries = []

libgalahad_c_single_src = []
libgalahad_c_double_src = []
libgalahad_c_integer_src = []
libgalahad_c_src = []

libgalahad_cpp_src = []
libgalahad_cutest_src = []
libgalahad_ampl_src = []
libgalahad_python_src = []

galahad_examples = []
galahad_c_examples = []
galahad_tests = []
galahad_c_tests = []
galahad_python_tests = []

# Headers and Fortran modules
libgalahad_include = [include_directories('include'),
                      include_directories('src/dum/include'),
                      include_directories('src/ampl')] + libhsl_modules

# TODO: -DSPRAL_NO_SCHED_GETCPU | DSPRAL_HAVE_SCHED_GETCPU
extra_args = ['-DSPRAL_NO_SCHED_GETCPU']
if libhwloc.found()
  extra_args += '-DSPRAL_HAVE_HWLOC'
else
  extra_args += '-DSPRAL_NO_HWLOC'
endif
extra_args_single = extra_args + '-DGALAHAD_SINGLE' + '-DSPRAL_SINGLE'
extra_args_double = extra_args + '-DGALAHAD_DOUBLE' + '-DSPRAL_DOUBLE'

# Sources
subdir('bin')
subdir('include')
subdir('src/amd')
subdir('src/ampl')
subdir('src/arc')
subdir('src/band')
subdir('src/bgo')
subdir('src/blls')
subdir('src/bqp')
subdir('src/bqpb')
subdir('src/bsc')
subdir('src/buildspec')
subdir('src/ccqp')
subdir('src/cdqp')
subdir('src/check')
subdir('src/checkpoint')
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
subdir('src/miqr')
subdir('src/mop')
subdir('src/nlpt')
subdir('src/nls')
subdir('src/norms')
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
subdir('src/scaling')
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
subdir('src/symbols')
subdir('src/tools')
subdir('src/trans')
subdir('src/trb')
subdir('src/trs')
subdir('src/tru')
subdir('src/ugo')
subdir('src/uls')
subdir('src/userdata')
subdir('src/warm')
subdir('src/wcp')
subdir('src/zd11')

# Libraries
if build_single
  gen_single = generator(fc_compiler,
                         output : 'single_@BASENAME@.f90',
                         arguments : pp_options + ['-DGALAHAD_SINGLE', '-DSPRAL_SINGLE',
                                     '-I', '@CURRENT_SOURCE_DIR@/include',
                                     '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                     '@INPUT@', '-o', '@OUTPUT@'])

  pp_sources_single = gen_single.process([libgalahad_src, libgalahad_c_src, libgalahad_cutest_src, libgalahad_ampl_src])

  sources_single = [libgalahad_multi_src, libgalahad_single_src, libgalahad_integer_src,
                    libgalahad_c_single_src, libgalahad_c_integer_src,
                    pp_sources_single, libgalahad_cpp_src, libgalahad_aux]

  libgalahad_single = library('galahad_single',
                              sources : sources_single,
                              dependencies : libgalahad_single_deps + libgalahad_deps,
                              fortran_args : extra_args_single,
                              cpp_args : extra_args_single,
                              link_language : 'fortran',
                              link_args : '-lstdc++',
                              include_directories: libgalahad_include,
                              install : true)
endif

if build_double
  gen_double = generator(fc_compiler,
                         output : 'double_@BASENAME@.f90',
                         arguments : pp_options + ['-DGALAHAD_DOUBLE', '-DSPRAL_DOUBLE',
                                     '-I', '@CURRENT_SOURCE_DIR@/include',
                                     '-I', '@CURRENT_SOURCE_DIR@/src/dum/include',
                                     '@INPUT@', '-o', '@OUTPUT@'])

  pp_sources_double = gen_double.process([libgalahad_src, libgalahad_c_src, libgalahad_cutest_src, libgalahad_ampl_src])

  sources_double = [libgalahad_multi_src, libgalahad_double_src, libgalahad_integer_src,
                    libgalahad_c_double_src, libgalahad_c_integer_src,
                    pp_sources_double, libgalahad_cpp_src, libgalahad_aux]

  libgalahad_double = library('galahad_double',
                              sources : sources_double,
                              dependencies : libgalahad_double_deps + libgalahad_deps,
                              fortran_args : extra_args_double,
                              cpp_args : extra_args_double,
                              link_language : 'fortran',
                              link_args : '-lstdc++',
                              include_directories: libgalahad_include,
                              install : true)
endif

if build_pythoniface and build_ciface and build_double
  incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check : true
  ).stdout().strip()
  libgalahad_include += [incdir_numpy]
  
  foreach interface: libgalahad_python_src
    name = interface[0]
    file = interface[1]

    libgalahad_python = py.extension_module(name,
                                            sources : file,
                                            link_args: ['-Wl,-Bsymbolic', '-Wl,-Bsymbolic-functions'],
                                            link_with : libgalahad_double,
                                            link_language : 'c',
                                            include_directories : libgalahad_include,
                                            subdir : 'galahad',
                                            install : true)
  endforeach
endif

# Binaries
foreach binary: galahad_binaries
  binname = binary[0]
  binfile = binary[1]

  if build_single
    executable(binname+'_single', binfile,
               dependencies : libgalahad_single_deps + libgalahad_deps,
               fortran_args : extra_args_single,
               link_with : libgalahad_single,
               link_language : 'fortran',
               include_directories: libgalahad_include,
               install : true)
  endif
  if build_double
    executable(binname+'_double', binfile,
               dependencies : libgalahad_double_deps + libgalahad_deps,
               fortran_args : extra_args_double,
               link_with : libgalahad_double,
               link_language : 'fortran',
               include_directories: libgalahad_include,
               install : true)
  endif
endforeach

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

      deps_tests = libgalahad_deps
      if precision == 'single'
        deps_tests += libgalahad_single_deps
      endif
      if precision == 'double'
        deps_tests += libgalahad_double_deps
      endif

      if not (name == 'croti_single')
        test(name,
             executable(name, file, fortran_args : args_precision, link_with : libgalahad_precision, dependencies : deps_tests,
                                    link_language : 'fortran', include_directories: libgalahad_include , install : true,
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
           executable(name, file, c_args : args_precision, link_with : libgalahad_precision, dependencies : lm, link_language : 'c',
                                  include_directories : libgalahad_include , install : true, install_dir : c_tests_folder),
           suite : [package, precision, 'C'],
           is_parallel : false)
    endforeach
  endforeach
endif

# Python tests
if false
  # TO DO: Add a check to insure that the shared library for the Python interfaces
  # is installed in a relevant location.
  if build_tests and build_pythoniface and build_ciface and build_double

    python_tests_folder = 'tests/Python'

    foreach test: galahad_python_tests
      package = test[0]
      name = test[1]
      file = test[2]
      test(name,
           py,
           args : file,
           suite : [package, 'Python'],
           is_parallel : false)
    endforeach
  endif
endif

# Fortran examples
if build_examples and build_double

  fortran_examples_folder = 'examples/Fortran'

  foreach example: galahad_examples
    name = example[0]
    file = example[1]
    executable(name, file, link_with : libgalahad_double, dependencies : libgalahad_deps, link_language : 'fortran',
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
      executable(name, file, c_args : args_precision, link_with : libgalahad_precision, dependencies : lm, link_language : 'c',
                             include_directories : libgalahad_include, install : true, install_dir : c_examples_folder)
    endforeach
  endforeach
endif
