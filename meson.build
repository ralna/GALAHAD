project(
  'GALAHAD',
  'c', 'cpp', 'fortran',
  version: '5.0.0',
  license: 'BSD-3',
  meson_version: '>= 0.61.0',
  default_options: [
    'buildtype=debug',
    'libdir=lib',
    'default_library=both',
    'warning_level=0',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
fc = meson.get_compiler('fortran')
fs = import('fs')

extra_args = []
if fc.get_id() == 'nagfor'
  extra_args += '-fixed'
  extra_args += '-dcfuns'
endif

# Preprocessing
if fc.get_id() == 'gcc'
  add_global_arguments('-cpp', language : 'fortran')
elif fc.get_id() == 'intel' or fc.get_id() == 'nagfor'
  add_global_arguments('-fpp', language : 'fortran')
elif fc.get_id() == 'flang'
  add_global_arguments('-c', language : 'fortran')
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

libomp = dependency('openmp', language : 'fortran', required : true)
libmpi = dependency('mpi', language : 'fortran', required : false)
libhwloc = dependency('hwloc', required : false)

libgalahad_single_deps = [libsmumps]
libgalahad_double_deps = [libdmumps]
libgalahad_deps = [libblas, liblapack, libmetis, libhsl, libcutest,
                   libwsmp, libumfpack, libpardiso, libspmf, libpastix,
                   libmkl_pardiso, libampl, libomp, libmpi, libhwloc]

libgalahad_single_src = []
libgalahad_double_src = []
libgalahad_integer_src = []
libgalahad_src = []

galahad_headers = []
galahad_binaries = []

libgalahad_c_single_src = []
libgalahad_c_double_src = []
libgalahad_c_integer_src = []
libgalahad_c_src = []

libgalahad_cutest_src = []
libgalahad_ampl_src = []

galahad_examples = []
galahad_c_examples = []
galahad_tests = []
galahad_c_tests = []

# Headers and Fortran modules
libgalahad_include = ['include', 'src/dum/include', 'src/ampl'] + libhsl_modules

# TODO: -DSPRAL_NO_SCHED_GETCPU | DSPRAL_HAVE_SCHED_GETCPU
extra_args = ['-DSPRAL_NO_SCHED_GETCPU']
if libhwloc.found()
  extra_args += '-DSPRAL_HAVE_HWLOC'
else
  extra_args += '-DSPRAL_NO_HWLOC'
endif
extra_args_single = extra_args + '-DGALAHAD_SINGLE' + '-DSPRAL_SINGLE'
extra_args_double = extra_args + '-DGALAHAD_DOUBLE' + '-DSPRAL_DOUBLE'

subdir('bin')
subdir('include')

## 1. Module requires for single and double precision
subdir('src/kinds')
subdir('src/buildspec')
subdir('src/lsrb')

## 2. the following modules have no dependencies inside libgalahad; they appear in alphabetical order
subdir('src/zd11')
subdir('src/band')
subdir('src/clock')
subdir('src/checkpoint')
subdir('src/common')
subdir('src/dum')
subdir('src/copyright')
subdir('src/extend')
subdir('src/external')
subdir('src/lapack')
subdir('src/lmt')
subdir('src/norms')
subdir('src/rand')
subdir('src/scu')
subdir('src/sort')
subdir('src/string')
subdir('src/symbols')
subdir('src/tools')
subdir('src/userdata')

## 3. the following modules have dependencies
## Order matters!
subdir('src/ampl')
subdir('src/smt')
subdir('src/convert')
subdir('src/qpt')
subdir('src/space')
subdir('src/specfile')
subdir('src/bsc')
subdir('src/fit')
subdir('src/hash')
subdir('src/roots')
subdir('src/gltr')
subdir('src/glrt')
subdir('src/lstr')
subdir('src/lsrt')
subdir('src/l2rt')
subdir('src/lhs')
subdir('src/lms')
subdir('src/presolve')
subdir('src/rpd')
subdir('src/sec')
subdir('src/sha')
subdir('src/nlpt')
subdir('src/ugo')
subdir('src/amd')
subdir('src/mop')
subdir('src/check')
subdir('src/fdh')
subdir('src/filter')
subdir('src/lqr')
subdir('src/lqt')
subdir('src/lsp')
subdir('src/miqr')
subdir('src/opt')
subdir('src/trans')
subdir('src/qpp')
subdir('src/scale')
subdir('src/scaling')
subdir('src/gls')
subdir('src/sils')
subdir('src/sls')
subdir('src/uls')
subdir('src/sbls')
subdir('src/lls')
subdir('src/ir')
subdir('src/trs')
subdir('src/icfs')
subdir('src/lpqp')
subdir('src/fdc')
subdir('src/cro')
subdir('src/qpd')
subdir('src/slls')
subdir('src/wcp')
subdir('src/problem')
subdir('src/rqs')
subdir('src/blls')
subdir('src/bqp')
subdir('src/cqp')
subdir('src/dqp')
subdir('src/dlp')
subdir('src/dps')
subdir('src/eqp')
subdir('src/demo')
subdir('src/l1qp')
subdir('src/psls')

subdir('src/lancelot')
subdir('src/lancelot_simple')
subdir('src/filtrane')
subdir('src/trb')
subdir('src/tru')
subdir('src/bgo')
subdir('src/nls')
subdir('src/arc')
subdir('src/dgo')
subdir('src/bqpb')
subdir('src/ccqp')
subdir('src/cdqp')
subdir('src/llst')
subdir('src/lpa')
subdir('src/lpb')
subdir('src/lsqp')
subdir('src/qpa')
subdir('src/qpb')
subdir('src/qpc')
subdir('src/qp')
subdir('src/warm')
subdir('src/ptrans')
subdir('src/cutest_functions')
subdir('src/ssids')
subdir('src/spral')

# Libraries
if build_single
  sources_single = [libgalahad_single_src, libgalahad_integer_src, libgalahad_src,
                    libgalahad_c_single_src, libgalahad_c_integer_src, libgalahad_c_src,
                    libgalahad_cutest_src, libgalahad_ampl_src]

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
  sources_double = [libgalahad_double_src, libgalahad_integer_src, libgalahad_src,
                    libgalahad_c_double_src, libgalahad_c_integer_src, libgalahad_c_src,
                    libgalahad_cutest_src, libgalahad_ampl_src]

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

# Binaries
foreach val: galahad_binaries
  binname = val[0]
  binfile = val[1]

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
      if not (name == 'croti_single')
        test(name,
             executable(name, file, fortran_args : args_precision, link_with : libgalahad_precision, dependencies : libgalahad_deps,
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
