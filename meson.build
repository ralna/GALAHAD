project(
  'GALAHAD',
  'c',
  'cpp',
  'fortran',
  version: '5.0.0',
  license: 'BSD-3',
  meson_version: '>= 0.61.0',
  default_options: [
    'c_std=c99',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
fc = meson.get_compiler('fortran')
fs = import('fs')  # filesystem module

extra_args = []
if fc.get_id() == 'nagfor'
  extra_args += '-fixed'
  extra_args += '-dcfuns'
endif

# Options
build_ciface = get_option('ciface')
build_pythoniface = get_option('pythoniface')
build_single = get_option('single')
build_double = get_option('double')
build_tests = get_option('tests')
build_examples = get_option('examples')
build_ssids = get_option('ssids')

libgalahad_link_with = []
libgalahad_include = ['include']
libgalahad_src = []
libgalahad_c_src = []
libgalahad_cpp_src = []
libgalahad_blas_src = []
libgalahad_lapack_src = []
subdir('src/lapack')  # libgalahad_blas/lapack only contain the.F90 interfaces
libgalahad_deps = []
blas_dep = []
lapack_dep = []

# -DSPRAL_HAVE_HWLOC | -DSPRAL_NO_HWLOC
# -DSPRAL_NO_SCHED_GETCPU | DSPRAL_HAVE_SCHED_GETCPU
extra_args_single = ['-DGALAHAD_SINGLE', '-DSPRAL_SINGLE', '-DSPRAL_NO_HWLOC', '-DSPRAL_NO_SCHED_GETCPU']
extra_args_double = ['-DGALAHAD_DOUBLE', '-DSPRAL_DOUBLE', '-DSPRAL_NO_HWLOC', '-DSPRAL_NO_SCHED_GETCPU']

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

# Decide what goes into libgalahad_hsl, if anything, based on information
# supplied by the user

libgalahad_hsl_single_src = []
libgalahad_hsl_double_src = []

libgalahad_hsl_single_src += files('src/kinds/kinds.F90')
libgalahad_hsl_double_src += files('src/kinds/kinds.F90')

has_hslarchive = false
hslarchive_galahad = get_option('hslarchive-galahad')
if hslarchive_galahad != ''
  if fs.is_dir(hslarchive_galahad)
    ad02_path = hslarchive_galahad / 'ad02'
    ma27_path = hslarchive_galahad / 'ma27'
    ma33_path = hslarchive_galahad / 'ma33'
    has_hslarchive = true
  else
    error('directory not found: ', hslarchive_galahad)
  endif
else
  warning('consider downloading hslarchive-galahad to improve functionality')
  ad02_path = 'src/dum'
  ma27_path = 'src/dum'
  ma33_path = 'src/dum'
endif

libgalahad_hsl_single_src += files(ad02_path / 'hsl_ad02s.f90')
libgalahad_hsl_double_src += files(ad02_path / 'hsl_ad02d.f90')

libgalahad_hsl_single_src += files(ma27_path / 'ma27s.f')
libgalahad_hsl_double_src += files(ma27_path / 'ma27d.f')

libgalahad_hsl_single_src += files(ma33_path / 'ma33s.f', ma33_path / 'mc13s.f', ma33_path / 'mc20s.f', ma33_path / 'mc21s.f')
libgalahad_hsl_double_src += files(ma33_path / 'ma33d.f', ma33_path / 'mc13d.f', ma33_path / 'mc20d.f', ma33_path / 'mc21d.f')

# FIXME: zd11 is in both libraries (libgalahad and libgalahad_hsl)
libgalahad_hsl_single_src += files('src/zd11/zd11.F90')
libgalahad_hsl_double_src += files('src/zd11/zd11.F90')

libgalahad_hsl_single_src += files('src/symbols/symbols.F90')
libgalahad_hsl_double_src += files('src/symbols/symbols.F90')

# FIXME
la04_path = 'src/dum'
ma61_path = 'src/dum'
mc13_path = 'src/dum'
mc21_path = 'src/dum'
mc23_path = 'src/dum'
mc61_path = 'src/dum'
mc77_path = 'src/dum'
hsl_of01_path = 'src/dum'
hsl_zb01_path = 'src/dum'
hsl_mi28_path = 'src/dum'
hsl_mc34_path = 'src/dum'
hsl_mc64_path = 'src/dum'
hsl_mc68_path = 'src/dum'
hsl_mc78_path = 'src/dum'
hsl_ma48_path = 'src/dum'
hsl_ma57_path = 'src/dum'
hsl_ma77_path = 'src/dum'
hsl_ma86_path = 'src/dum'
hsl_ma87_path = 'src/dum'
hsl_ma97_path = 'src/dum'

libgalahad_hsl_single_src += files(la04_path / 'la04s.f')
libgalahad_hsl_double_src += files(la04_path / 'la04d.f')

libgalahad_hsl_single_src += files(ma61_path / 'ma61s.f')
libgalahad_hsl_double_src += files(ma61_path / 'ma61d.f')

libgalahad_hsl_single_src += files(mc13_path / 'mc13s.f')
libgalahad_hsl_double_src += files(mc13_path / 'mc13d.f')

libgalahad_hsl_single_src += files(mc21_path / 'mc21s.f')
libgalahad_hsl_double_src += files(mc21_path / 'mc21d.f')

libgalahad_hsl_single_src += files(mc23_path / 'mc23s.f')
libgalahad_hsl_double_src += files(mc23_path / 'mc23d.f')

libgalahad_hsl_single_src += files(mc61_path / 'mc61s.f')
libgalahad_hsl_double_src += files(mc61_path / 'mc61d.f')

libgalahad_hsl_single_src += files(mc77_path / 'mc77s.f')
libgalahad_hsl_double_src += files(mc77_path / 'mc77d.f')

libgalahad_hsl_single_src += files(hsl_of01_path / 'hsl_of01s.f90', hsl_of01_path / 'hsl_of01i.f90')
libgalahad_hsl_double_src += files(hsl_of01_path / 'hsl_of01d.f90', hsl_of01_path / 'hsl_of01i.f90')

libgalahad_hsl_single_src += files(hsl_zb01_path / 'hsl_zb01s.f90', hsl_zb01_path / 'hsl_zb01i.f90')
libgalahad_hsl_double_src += files(hsl_zb01_path / 'hsl_zb01d.f90', hsl_zb01_path / 'hsl_zb01i.f90')

libgalahad_hsl_single_src += files(hsl_mc34_path / 'hsl_mc34s.f90')
libgalahad_hsl_double_src += files(hsl_mc34_path / 'hsl_mc34d.f90')

libgalahad_hsl_single_src += files(hsl_mc64_path / 'hsl_mc64s.f90')
libgalahad_hsl_double_src += files(hsl_mc64_path / 'hsl_mc64d.f90')

libgalahad_hsl_single_src += files(hsl_mc68_path / 'hsl_mc68s.f90', hsl_mc68_path / 'hsl_mc68i.f90')
libgalahad_hsl_double_src += files(hsl_mc68_path / 'hsl_mc68d.f90', hsl_mc68_path / 'hsl_mc68i.f90')

libgalahad_hsl_single_src += files(hsl_mc78_path / 'hsl_mc78i.f90')
libgalahad_hsl_double_src += files(hsl_mc78_path / 'hsl_mc78i.f90')

libgalahad_hsl_single_src += files(hsl_ma48_path / 'hsl_ma48s.f90')
libgalahad_hsl_double_src += files(hsl_ma48_path / 'hsl_ma48d.f90')

libgalahad_hsl_single_src += files(hsl_ma57_path / 'hsl_ma57s.f90')
libgalahad_hsl_double_src += files(hsl_ma57_path / 'hsl_ma57d.f90')

libgalahad_hsl_single_src += files(hsl_ma77_path / 'hsl_ma77s.f90')
libgalahad_hsl_double_src += files(hsl_ma77_path / 'hsl_ma77d.f90')

libgalahad_hsl_single_src += files(hsl_ma86_path / 'hsl_ma86s.f90')
libgalahad_hsl_double_src += files(hsl_ma86_path / 'hsl_ma86d.f90')

libgalahad_hsl_single_src += files(hsl_ma87_path / 'hsl_ma87s.f90')
libgalahad_hsl_double_src += files(hsl_ma87_path / 'hsl_ma87d.f90')

libgalahad_hsl_single_src += files(hsl_ma97_path / 'hsl_ma97s.f90')
libgalahad_hsl_double_src += files(hsl_ma97_path / 'hsl_ma97d.f90')

libgalahad_hsl_single_src += files(hsl_mi28_path / 'hsl_mi28s.f90')
libgalahad_hsl_double_src += files(hsl_mi28_path / 'hsl_mi28d.f90')

libgalahad_hsl_single = library('galahad_hsl_single',
                                 sources : libgalahad_hsl_single_src,
                                 link_with : [libgalahad_blas, libgalahad_lapack],
                                 dependencies : blas_dep + lapack_dep,
                                 fortran_args : extra_args_single,
                                 include_directories: libgalahad_include,
                                 install : true)

libgalahad_hsl_double = library('galahad_hsl_double',
                                sources : libgalahad_hsl_double_src,
                                link_with : [libgalahad_blas, libgalahad_lapack],
                                dependencies : blas_dep + lapack_dep,
                                fortran_args : extra_args_double,
                                include_directories: libgalahad_include,
                                install : true
)
libmkl_pardiso_src = []
libpardiso_src = []
libspral_ssids_single_src = []
libspral_ssids_double_src = []
libmumps_src = []
libspmf_src = []
libpastix_src = []
libwsmp_src = []

subdir('src/external')

libmkl_pardiso_src += files('src/kinds/kinds.F90', 'src/symbols/symbols.F90', 'src/dum/mkl_pardiso.F90')

libmkl_pardiso = library('mkl_pardiso',
                         sources : libmkl_pardiso_src,
                         link_with : [libgalahad_blas, libgalahad_lapack],
                         dependencies : blas_dep + lapack_dep,
                         install : true)

libpardiso_src += files('src/kinds/kinds.F90', 'src/symbols/symbols.F90', 'src/dum/pardiso.F90')

libpardiso = library('pardiso',
                     sources : libpardiso_src,
                     link_with : [libgalahad_blas, libgalahad_lapack],
                     dependencies : blas_dep + lapack_dep,
                     include_directories: libgalahad_include,
                     install : true)

libspral_ssids_single_src += files('src/dum/ssidss.F90', 'src/kinds/kinds.F90', 'src/spral/kinds.F90', 'src/symbols/symbols.F90')
libspral_ssids_double_src += files('src/dum/ssidsd.F90', 'src/kinds/kinds.F90', 'src/spral/kinds.F90', 'src/symbols/symbols.F90')

libspral_ssids_single = library('spral_ssids_single',
                                sources : libspral_ssids_single_src,
                                link_with : [libgalahad_blas, libgalahad_lapack],
                                dependencies : blas_dep + lapack_dep,
                                install : true)

libspral_ssids_double = library('spral_ssids_double',
                                sources : libspral_ssids_double_src,
                                link_with : [libgalahad_blas, libgalahad_lapack],
                                dependencies : blas_dep + lapack_dep,
                                install : true)

libmumps_src += files('src/kinds/kinds.F90', 'src/dum/mpi.F90', 'src/dum/mumps.F90')

libmumps_single = library('mumps_single',
                          sources : libmumps_src,
                          link_with : [libgalahad_blas, libgalahad_lapack],
                          dependencies : blas_dep + lapack_dep,
                          fortran_args : extra_args_single,
                          include_directories: libgalahad_include + 'src/dum/include',
                          install : true)

libmumps_double = library('mumps_double',
                          sources : libmumps_src,
                          link_with : [libgalahad_blas, libgalahad_lapack],
                          dependencies : blas_dep + lapack_dep,
                          fortran_args : extra_args_double,
                          include_directories: libgalahad_include + 'src/dum/include',
                          install : true)

libspmf_src += files('src/kinds/kinds.F90', 'src/dum/spmf_enums.F90', 'src/external/pastix/spmf_interfaces.F90', 'src/dum/spmf.F90')

libspmf_single = library('spmf_single',
                         sources : libspmf_src,
                         link_with : [libgalahad_blas, libgalahad_lapack],
                         dependencies : blas_dep + lapack_dep,
                         fortran_args : extra_args_single,
                         install : true)

libspmf_double = library('spmf_double',
                         sources : libspmf_src,
                         link_with : [libgalahad_blas, libgalahad_lapack],
                         dependencies : blas_dep + lapack_dep,
                         fortran_args : extra_args_double,
                         install : true)

libpastix_src += files('src/kinds/kinds.F90', 'src/dum/pastixf_enums.F90', 'src/external/pastix/pastixf_interfaces.F90', 'src/dum/pastixf.F90')

libpastix_single = library('pastix_single',
                    sources : libpastix_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libspmf_single],
                    dependencies : blas_dep + lapack_dep,
                    fortran_args : extra_args_single,
                    install : true)

libpastix_double = library('pastix_double',
                    sources : libpastix_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libspmf_double],
                    dependencies : blas_dep + lapack_dep,
                    fortran_args : extra_args_double,
                    install : true)

libwsmp_src += files('src/kinds/kinds.F90', 'src/symbols/symbols.F90', 'src/dum/wsmp.F90')

libwsmp = library('wsmp',
                  sources : libwsmp_src,
                  link_with : [libgalahad_blas, libgalahad_lapack],
                  dependencies : blas_dep + lapack_dep,
                  include_directories: libgalahad_include,
                  install : true)

libgalahad_metis4 = library('galahad_metis4',
                           sources : 'src/dum/metis4.f',
                           install : true)

## 1. Module requires for single and double precision
subdir('src/kinds')

## 2. the following modules have no dependencies inside libgalahad; they appear in alphabetical order
subdir('src/band')
subdir('src/clock')
subdir('src/checkpoint')
subdir('src/common')
subdir('src/dum')
subdir('src/copyright')
subdir('src/extend')
subdir('src/lmt')
subdir('src/norms')
subdir('src/rand')
subdir('src/scu')
subdir('src/sort')
subdir('src/string')
subdir('src/symbols')
subdir('src/tools')
subdir('src/userdata')
subdir('src/zd11')

## 3. the following modules have dependencies
## Order matters!
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

# needs CUTEst interface
# subdir('src/ptrans')
# subdir('src/cutest_functions')

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

# needs c++ files
subdir('src/spral')
subdir('src/ssids')
# Add an option gpu

libgalahad_cpp_single = library('galahad_cpp_single',
                                 sources : libgalahad_cpp_src,
                                 link_with : [libgalahad_blas, libgalahad_lapack],
                                 dependencies : blas_dep + lapack_dep,
                                 cpp_args : extra_args_single,
                                 # link_language : 'cpp',
                                 include_directories: libgalahad_include,
                                 install : true)

libgalahad_cpp_double = library('galahad_cpp_double',
                                sources : libgalahad_cpp_src,
                                link_with : [libgalahad_blas, libgalahad_lapack],
                                dependencies : blas_dep + lapack_dep,
                                cpp_args : extra_args_double,
                                # link_language : 'cpp',
                                include_directories: libgalahad_include,
                                install : true)

if build_single

  libgalahad_single_link_with = [libgalahad_blas, libgalahad_lapack, libgalahad_hsl_single,
                                 libpardiso, libmkl_pardiso, libmumps_single, libspmf_single,
                                 libwsmp, libpastix_single, libgalahad_metis4, libgalahad_cpp_single]

  if not build_ssids
    libspral_ssids_single = library('spral_ssids_single',
                                    sources : libspral_ssids_single_src,
                                    link_with : [libgalahad_blas, libgalahad_lapack],
                                    dependencies : blas_dep + lapack_dep,
                                    # link_language : 'cpp',
                                    install : true)

    libgalahad_single_link_with += libspral_ssids_single
  endif

  libgalahad_single = library('galahad_single',
                              sources : libgalahad_src,
                              link_with : libgalahad_single_link_with,
                              dependencies : libgalahad_deps,
                              fortran_args : extra_args_single,
                              cpp_args : extra_args_single,
                              # link_language : 'fortran',
                              include_directories: libgalahad_include,
                              install : true)

  if build_ciface

    libgalahad_single_c_link_with = [libgalahad_single, libgalahad_hsl_single]
    if not build_ssids
      libgalahad_single_c_link_with += libspral_ssids_single
    endif

    libgalahad_c_single = library('galahad_c_single',
                                  sources : dum_single_c_src + libgalahad_c_src,
                                  link_with : libgalahad_single_c_link_with,
                                  fortran_args : extra_args_single,
                                  # link_language : 'fortran',
                                  include_directories: libgalahad_include,
                                  install : true)
  endif
endif

if build_double

  libgalahad_double_link_with = [libgalahad_blas, libgalahad_lapack, libgalahad_hsl_double,
                                 libpardiso, libmkl_pardiso, libmumps_double, libspmf_double,
                                 libwsmp, libpastix_double, libgalahad_metis4, libgalahad_cpp_double]

  if not build_ssids
    libspral_ssids_double = library('spral_ssids_double',
                                    sources : libspral_ssids_double_src,
                                    link_with : [libgalahad_blas, libgalahad_lapack],
                                    dependencies : blas_dep + lapack_dep,
                                    # link_language : 'cpp',
                                    install : true)

    libgalahad_double_link_with += libspral_ssids_double
  endif

  libgalahad_double = library('galahad_double',
                              sources : libgalahad_src,
                              link_with : libgalahad_double_link_with,
                              dependencies : libgalahad_deps,
                              fortran_args : extra_args_double,
                              cpp_args : extra_args_double,
                              # link_language : 'fortran',
                              include_directories: libgalahad_include,
                              install : true)

  if build_ciface

    libgalahad_double_c_link_with = [libgalahad_double, libgalahad_hsl_double]
    if not build_ssids
      libgalahad_double_c_link_with += libspral_ssids_double
    endif

    libgalahad_c_double = library('galahad_c_double',
                                  sources : dum_double_c_src  + libgalahad_c_src,
                                  link_with : libgalahad_double_c_link_with,
                                  fortran_args : extra_args_double,
                                  # link_language : 'fortran',
                                  include_directories: libgalahad_include,
                                  install : true)
  endif
endif

# Executables
if build_single
  buildspec_single = executable('buildspec_single', 'src/buildspec/buildspec.F90',
                                fortran_args : extra_args_single,
                                link_with : libgalahad_single,
                                link_language : 'fortran',
                                include_directories: libgalahad_include,
                                install : true)
endif
if build_double
  buildspec_double = executable('buildspec_double', 'src/buildspec/buildspec.F90',
                                fortran_args : extra_args_double,
                                link_with : libgalahad_double,
                                link_language : 'fortran',
                                include_directories: libgalahad_include,
                                install : true)
endif

# Unit tests

if build_tests

  fortran_tests_folder = 'share/tests/fortran'
  fortran_tests = []
  if build_single
    fortran_tests += [['single', extra_args_single, libgalahad_single, libmumps_single, libpastix_single, libspmf_single]]
  endif
  if build_double
    fortran_tests += [['double', extra_args_double, libgalahad_double, libmumps_double, libpastix_double, libspmf_double]]
  endif

  foreach val: fortran_tests
    amdt = executable('amdt_'+val[0], 'src/amd/amdt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('amdt_'+val[0], amdt, suite: ['amd', 'fortran', val[0]], is_parallel : false)

    bsct = executable('bsct_'+val[0], 'src/bsc/bsct.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('bsct_'+val[0], bsct, suite: ['bsc', 'fortran', val[0]], is_parallel : false)

    checkt = executable('checkt_'+val[0], 'src/check/checkt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('checkt_'+val[0], checkt, suite: ['check', 'fortran', val[0]], is_parallel : false)

    convertt = executable('convertt_'+val[0], 'src/convert/convertt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('convertt_'+val[0], convertt, suite: ['convert', 'fortran', val[0]], is_parallel : false)

    fdht = executable('fdht_'+val[0], 'src/fdh/fdht.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('fdht_'+val[0], fdht, suite: ['fdh', 'fortran', val[0]], is_parallel : false)

    fitt = executable('fitt_'+val[0], 'src/fit/fitt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('fitt_'+val[0], fitt, suite: ['fit', 'fortran', val[0]], is_parallel : false)

    gltrt = executable('gltrt_'+val[0], 'src/gltr/gltrt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    gltrti = executable('gltrti_'+val[0], 'src/gltr/gltrti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('gltrt_'+val[0], gltrt, suite: ['gltr', 'fortran', val[0]], is_parallel : false)
    test('gltrti_'+val[0], gltrti, suite: ['gltr', 'fortran', val[0]], is_parallel : false)

    dgot = executable('dgot_'+val[0], 'src/dgo/dgot.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    dgoti = executable('dgoti_'+val[0], 'src/dgo/dgoti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('dgot_'+val[0], dgot, suite: ['dgo', 'fortran', val[0]], is_parallel : false)
    test('dgoti_'+val[0], dgoti, suite: ['dgo', 'fortran', val[0]], is_parallel : false)

    glrtt = executable('glrtt_'+val[0], 'src/glrt/glrtt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('glrtt_'+val[0], glrtt, suite: ['glrt', 'fortran', val[0]], is_parallel : false)

    glst = executable('glst_'+val[0], 'src/gls/glst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('glst_'+val[0], glst, suite: ['gls', 'fortran', val[0]], is_parallel : false)

    crot = executable('crot_'+val[0], 'src/cro/crot.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('crot_'+val[0], crot, suite: ['cro', 'fortran', val[0]], is_parallel : false)

    hasht = executable('hasht_'+val[0], 'src/hash/hasht.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('hasht_'+val[0], hasht, suite: ['hash', 'fortran', val[0]], is_parallel : false)

    lqrt = executable('lqrt_'+val[0], 'src/lqr/lqrt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lqrt_'+val[0], lqrt, suite: ['lqr', 'fortran', val[0]], is_parallel : false)

    lancelot_simplet = executable('lancelot_simplet_'+val[0], 'src/lancelot_simple/lancelot_simplet.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lancelot_simplet_'+val[0], lancelot_simplet, suite: ['lancelot_simple', 'fortran', val[0]], is_parallel : false)

    lqtt = executable('lqtt_'+val[0], 'src/lqt/lqtt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lqtt2 = executable('lqtt2_'+val[0], 'src/lqt/lqtt2.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lqtt_'+val[0], lqtt, suite: ['lqt', 'fortran', val[0]], is_parallel : false)
    test('lqtt2_'+val[0], lqtt2, suite: ['lqt', 'fortran', val[0]], is_parallel : false)

    lspt = executable('lspt_'+val[0], 'src/lsp/lspt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lspt_'+val[0], lspt, suite: ['lsp', 'fortran', val[0]], is_parallel : false)

    lstrt = executable('lstrt_'+val[0], 'src/lstr/lstrt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lstrti = executable('lstrti_'+val[0], 'src/lstr/lstrti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lstrt_'+val[0], lstrt, suite: ['lstr', 'fortran', val[0]], is_parallel : false)
    test('lstrti_'+val[0], lstrti, suite: ['lstr', 'fortran', val[0]], is_parallel : false)

    fdct = executable('fdct_'+val[0], 'src/fdc/fdct.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    fdcti = executable('fdcti_'+val[0], 'src/fdc/fdcti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('fdct_'+val[0], fdct, suite: ['fdc', 'fortran', val[0]], is_parallel : false)
    test('fdcti_'+val[0], fdcti, suite: ['fdc', 'fortran', val[0]], is_parallel : false)

    lsrtt = executable('lsrtt_'+val[0], 'src/lsrt/lsrtt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lsrtti = executable('lsrtti_'+val[0], 'src/lsrt/lsrtti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lsrtt_'+val[0], lsrtt, suite: ['lsrt', 'fortran', val[0]], is_parallel : false)
    test('lsrtti_'+val[0], lsrtti, suite: ['lsrt', 'fortran', val[0]], is_parallel : false)

    l2rtt  = executable('l2rtt_'+val[0] , 'src/l2rt/l2rtt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    l2rtti = executable('l2rtti_'+val[0], 'src/l2rt/l2rtti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('l2rtt_'+val[0], l2rtt, suite: ['l2rt', 'fortran', val[0]], is_parallel : false)
    test('l2rtti_'+val[0], l2rtti, suite: ['l2rt', 'fortran', val[0]], is_parallel : false)

    rqst  = executable('rqst_'+val[0] , 'src/rqs/rqst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    rqsti = executable('rqsti_'+val[0], 'src/rqs/rqsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('rqst_'+val[0], rqst, suite: ['rqs', 'fortran', val[0]], is_parallel : false)
    test('rqsti_'+val[0], rqsti, suite: ['rqs', 'fortran', val[0]], is_parallel : false)

    lhst = executable('lhst_'+val[0], 'src/lhs/lhst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lhst_'+val[0], lhst, suite: ['lhs', 'fortran', val[0]], is_parallel : false)

    lmst = executable('lmst_'+val[0], 'src/lms/lmst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lmst_'+val[0], lmst, suite: ['lms', 'fortran', val[0]], is_parallel : false)

    miqrt = executable('miqrt_'+val[0], 'src/miqr/miqrt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('miqrt_'+val[0], miqrt, suite: ['miqr', 'fortran', val[0]], is_parallel : false)

    mopt = executable('mopt_'+val[0], 'src/mop/mopt.F90', fortran_args : val[1], link_with : [val[2], libgalahad_blas], link_language: 'fortran', include_directories: libgalahad_include)
    test('mopt_'+val[0], mopt, suite: ['mop', 'fortran', val[0]], is_parallel : false)

    dpst = executable('dpst_'+val[0], 'src/dps/dpst.F90', fortran_args : val[1], link_with : [val[2], libgalahad_blas], link_language: 'fortran', include_directories: libgalahad_include)
    test('dpst_'+val[0], dpst, suite: ['dps', 'fortran', val[0]], is_parallel : false)

    sllst = executable('sllst_'+val[0], 'src/slls/sllst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    sllsti = executable('sllsti_'+val[0], 'src/slls/sllsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('sllst_'+val[0], sllst, suite: ['slls', 'fortran', val[0]], is_parallel : false)
    test('sllsti_'+val[0], sllsti, suite: ['slls', 'fortran', val[0]], is_parallel : false)

    nlptt = executable('nlptt_'+val[0], 'src/nlpt/nlptt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('nlptt_'+val[0], nlptt, suite: ['nlpt', 'fortran', val[0]], is_parallel : false)

    presolvet = executable('presolvet_'+val[0], 'src/presolve/presolvet.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    presolveti = executable('presolveti_'+val[0], 'src/presolve/presolveti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('presolvet_'+val[0], presolvet, suite: ['presolve', 'fortran', val[0]], is_parallel : false)
    test('presolveti_'+val[0], presolveti, suite: ['presolve', 'fortran', val[0]], is_parallel : false)

    qppt = executable('qppt_'+val[0], 'src/qpp/qppt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qppt_'+val[0], qppt, suite: ['qpp', 'fortran', val[0]], is_parallel : false)

    qptt = executable('qptt_'+val[0], 'src/qpt/qptt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qptt_'+val[0], qptt, suite: ['qpt', 'fortran', val[0]], is_parallel : false)

    randt = executable('randt_'+val[0], 'src/rand/randt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('randt_'+val[0], randt, suite: ['rand', 'fortran', val[0]], is_parallel : false)

    rpdt = executable('rpdt_'+val[0], 'src/rpd/rpdt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    rpdti = executable('rpdti_'+val[0], 'src/rpd/rpdti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('rpdt_'+val[0], rpdt, suite: ['rpd', 'fortran', val[0]], is_parallel : false)
    test('rpdti_'+val[0], rpdti, suite: ['rpd', 'fortran', val[0]], is_parallel : false)

    bllst = executable('bllst_'+val[0], 'src/blls/bllst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    bllsti = executable('bllsti_'+val[0], 'src/blls/bllsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('bllst_'+val[0], bllst, suite: ['blls', 'fortran', val[0]], is_parallel : false)
    test('bllsti_'+val[0], bllsti, suite: ['blls', 'fortran', val[0]], is_parallel : false)

    trut = executable('trut_'+val[0], 'src/tru/trut.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    truti = executable('truti_'+val[0], 'src/tru/truti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('trut_'+val[0], trut, suite: ['tru', 'fortran', val[0]], is_parallel : false)
    test('truti_'+val[0], truti, suite: ['tru', 'fortran', val[0]], is_parallel : false)

    rootst = executable('rootst_'+val[0], 'src/roots/rootst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('rootst_'+val[0], rootst, suite: ['roots', 'fortran', val[0]], is_parallel : false)

    scut = executable('scut_'+val[0], 'src/scu/scut.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('scut_'+val[0], scut, suite: ['scu', 'fortran', val[0]], is_parallel : false)

    scalet = executable('scalet_'+val[0], 'src/scale/scalet.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('scalet_'+val[0], scalet, suite: ['scale', 'fortran', val[0]], is_parallel : false)

    sect = executable('sect_'+val[0], 'src/sec/sect.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('sect_'+val[0], sect, suite: ['sec', 'fortran', val[0]], is_parallel : false)

    shat = executable('shat_'+val[0], 'src/sha/shat.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('shat_'+val[0], shat, suite: ['sha', 'fortran', val[0]], is_parallel : false)

    silst = executable('silst_'+val[0], 'src/sils/silst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('silst_'+val[0], silst, suite: ['sils', 'fortran', val[0]], is_parallel : false)

    smtt = executable('smtt_'+val[0], 'src/smt/smtt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('smtt_'+val[0], smtt, suite: ['smt', 'fortran', val[0]], is_parallel : false)

    sortt = executable('sortt_'+val[0], 'src/sort/sortt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('sortt_'+val[0], sortt, suite: ['sort', 'fortran', val[0]], is_parallel : false)

    ugot = executable('ugot_'+val[0], 'src/ugo/ugot.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('ugot_'+val[0], ugot, suite: ['ugo', 'fortran', val[0]], is_parallel : false)

    mkl_pardisot = executable('mkl_pardisot_'+val[0], 'src/external/mkl_pardiso/mkl_pardisot.F90', fortran_args : val[1], link_with : libmkl_pardiso, link_language: 'fortran', include_directories: libgalahad_include)
    test('mkl_pardisot_'+val[0], mkl_pardisot, suite: ['mkl_pardiso', 'fortran', val[0]], is_parallel : false)

    mumpst = executable('mumpst_'+val[0], 'src/external/mumps/mumpst.F90', fortran_args : val[1], link_with : val[3], link_language: 'fortran', include_directories: libgalahad_include)
    test('mumpst_'+val[0], mumpst, suite: ['mumps', 'fortran', val[0]], is_parallel : false)

    pardisot = executable('pardisot_'+val[0], 'src/external/pardiso/pardisot.F90', fortran_args : val[1], link_with : libpardiso, link_language: 'fortran', include_directories: libgalahad_include)
    test('pardisot_'+val[0], pardisot, suite: ['pardiso', 'fortran', val[0]], is_parallel : false)

    wsmpt = executable('wsmpt_'+val[0], 'src/external/wsmp/wsmpt.F90', fortran_args : val[1], link_with : libwsmp, link_language: 'fortran', include_directories: libgalahad_include)
    test('wsmpt_'+val[0], wsmpt, suite: ['wsmp', 'fortran', val[0]], is_parallel : false)

    pastixt = executable('pastixt_'+val[0], 'src/external/pastix/pastixt.F90', fortran_args : val[1], link_with : [val[4], val[5]], link_language: 'fortran', include_directories: libgalahad_include)
    test('pastixt_'+val[0], pastixt, suite: ['pastix', 'fortran', val[0]], is_parallel : false)

    slst = executable('slst_'+val[0], ['src/sls/slst.F90', 'src/dum/mpi.F90'], fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    slsti = executable('slsti_'+val[0], 'src/sls/slsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('slst_'+val[0], slst, suite: ['sls', 'fortran', val[0]], is_parallel : false)
    test('slsti_'+val[0], slsti, suite: ['sls', 'fortran', val[0]], is_parallel : false)

    ulst = executable('ulst_'+val[0], 'src/uls/ulst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    ulsti = executable('ulsti_'+val[0], 'src/uls/ulsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('ulst_'+val[0], ulst, suite: ['uls', 'fortran', val[0]], is_parallel : false)
    test('ulsti_'+val[0], ulsti, suite: ['uls', 'fortran', val[0]], is_parallel : false)

    irt = executable('irt_'+val[0], 'src/ir/irt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('ir_'+val[0], irt, suite: ['ir', 'fortran', val[0]], is_parallel : false)

    icfst = executable('icfst_'+val[0], 'src/icfs/icfst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('icfst_'+val[0], icfst, suite: ['icfs', 'fortran', val[0]], is_parallel : false)

    trst = executable('trst_'+val[0], 'src/trs/trst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    trsti = executable('trsti_'+val[0], 'src/trs/trsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('trst_'+val[0], trst, suite: ['trs', 'fortran', val[0]], is_parallel : false)
    test('trsti_'+val[0], trsti, suite: ['trs', 'fortran', val[0]], is_parallel : false)

    trbt = executable('trbt_'+val[0], 'src/trb/trbt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    trbti = executable('trbti_'+val[0], 'src/trb/trbti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('trbt_'+val[0], trbt, suite: ['trb', 'fortran', val[0]], is_parallel : false)
    test('trbti_'+val[0], trbti, suite: ['trb', 'fortran', val[0]], is_parallel : false)

    wcpt = executable('wcpt_'+val[0], 'src/wcp/wcpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    wcpti = executable('wcpti_'+val[0], 'src/wcp/wcpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('wcpt_'+val[0], wcpt, suite: ['wcp', 'fortran', val[0]], is_parallel : false)
    test('wcpti_'+val[0], wcpti, suite: ['wcp', 'fortran', val[0]], is_parallel : false)

    pslst = executable('pslst_'+val[0], 'src/psls/pslst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    pslsti = executable('pslsti_'+val[0], 'src/psls/pslsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('pslst_'+val[0], pslst, suite: ['psls', 'fortran', val[0]], is_parallel : false)
    test('pslsti_'+val[0], pslsti, suite: ['psls', 'fortran', val[0]], is_parallel : false)

    lsqpt = executable('lsqpt_'+val[0], 'src/lsqp/lsqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lsqpti = executable('lsqpti_'+val[0], 'src/lsqp/lsqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lsqpt_'+val[0], lsqpt, suite: ['lsqp', 'fortran', val[0]], is_parallel : false)
    test('lsqpti_'+val[0], lsqpti, suite: ['lsqp', 'fortran', val[0]], is_parallel : false)

    lpat = executable('lpat_'+val[0], 'src/lpa/lpat.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lpati = executable('lpati_'+val[0], 'src/lpa/lpati.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lpat_'+val[0], lpat, suite: ['lpa', 'fortran', val[0]], is_parallel : false)
    test('lpati_'+val[0], lpati, suite: ['lpa', 'fortran', val[0]], is_parallel : false)

    lpbt = executable('lpbt_'+val[0], 'src/lpb/lpbt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lpbti = executable('lpbti_'+val[0], 'src/lpb/lpbti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lpbt_'+val[0], lpbt, suite: ['lpb', 'fortran', val[0]], is_parallel : false)
    test('lpbti_'+val[0], lpbti, suite: ['lpb', 'fortran', val[0]], is_parallel : false)

    arct = executable('arct_'+val[0], 'src/arc/arct.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    arcti = executable('arcti_'+val[0], 'src/arc/arcti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('arct_'+val[0], arct, suite: ['arc', 'fortran', val[0]], is_parallel : false)
    test('arcti_'+val[0], arcti, suite: ['arc', 'fortran', val[0]], is_parallel : false)

    sblst = executable('sblst_'+val[0], 'src/sbls/sblst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    sblsti = executable('sblsti_'+val[0], 'src/sbls/sblsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('sblst_'+val[0], sblst, suite: ['sbls', 'fortran', val[0]], is_parallel : false)
    test('sblsti_'+val[0], sblsti, suite: ['sbls', 'fortran', val[0]], is_parallel : false)

    cqpt = executable('cqpt_'+val[0], 'src/cqp/cqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    cqpti = executable('cqpti_'+val[0], 'src/cqp/cqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('cqpt_'+val[0], cqpt, suite: ['cqp', 'fortran', val[0]], is_parallel : false)
    test('cqpti_'+val[0], cqpti, suite: ['cqp', 'fortran', val[0]], is_parallel : false)

    bqpt = executable('bqpt_'+val[0], 'src/bqp/bqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    bqpti = executable('bqpti_'+val[0], 'src/bqp/bqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('bqpt_'+val[0], bqpt, suite: ['bqp', 'fortran', val[0]], is_parallel : false)
    test('bqpti_'+val[0], bqpti, suite: ['bqp', 'fortran', val[0]], is_parallel : false)

    lancelott = executable('lancelott_'+val[0], 'src/lancelot/lancelott.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    lancelot_steeringt = executable('lancelot_steeringt_'+val[0], 'src/lancelot/lancelot_steeringt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('lancelott_'+val[0], lancelott, suite: ['lancelot', 'fortran', val[0]], is_parallel : false)
    test('lancelot_steeringt_'+val[0], lancelot_steeringt, suite: ['lancelot', 'fortran', val[0]], is_parallel : false)

    llstt = executable('llstt_'+val[0], 'src/llst/llstt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('llstt_'+val[0], llstt, suite: ['llst', 'fortran', val[0]], is_parallel : false)

    l1qpt = executable('l1qpt_'+val[0], 'src/l1qp/l1qpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('l1qpt_'+val[0], l1qpt, suite: ['llqp', 'fortran', val[0]], is_parallel : false)

    dlpt = executable('dlpt_'+val[0], 'src/dlp/dlpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('dlpt_'+val[0], dlpt, suite: ['qlp', 'fortran', val[0]], is_parallel : false)

    dqpt = executable('dqpt_'+val[0], 'src/dqp/dqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    dqpti = executable('dqpti_'+val[0], 'src/dqp/dqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('dqpt_'+val[0], dqpt, suite: ['dqp', 'fortran', val[0]], is_parallel : false)
    test('dqpti_'+val[0], dqpti, suite: ['dqp', 'fortran', val[0]], is_parallel : false)

    eqpt = executable('eqpt_'+val[0], 'src/eqp/eqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    eqpti = executable('eqpti_'+val[0], 'src/eqp/eqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('eqpt_'+val[0], eqpt, suite: ['eqp', 'fortran', val[0]], is_parallel : false)
    test('eqpti_'+val[0], eqpti, suite: ['eqp', 'fortran', val[0]], is_parallel : false)

    bgot = executable('bgot_'+val[0], 'src/bgo/bgot.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    bgoti = executable('bgoti_'+val[0], 'src/bgo/bgoti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('bgot_'+val[0], bgot, suite: ['bgo', 'fortran', val[0]], is_parallel : false)
    test('bgoti_'+val[0], bgoti, suite: ['bgo', 'fortran', val[0]], is_parallel : false)

    nlst = executable('nlst_'+val[0], 'src/nls/nlst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    nlsti = executable('nlsti_'+val[0], 'src/nls/nlsti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('nlst_'+val[0], nlst, suite: ['nls', 'fortran', val[0]], is_parallel : false)
    test('nlsti_'+val[0], nlsti, suite: ['nls', 'fortran', val[0]], is_parallel : false)

    bqpbt = executable('bqpbt_'+val[0], 'src/bqpb/bqpbt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    bqpbti = executable('bqpbti_'+val[0], 'src/bqpb/bqpbti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('bqpbt_'+val[0], bqpbt, suite: ['bqpb', 'fortran', val[0]], is_parallel : false)
    test('bqpbti_'+val[0], bqpbti, suite: ['bqpb', 'fortran', val[0]], is_parallel : false)

    ccqpt = executable('ccqpt_'+val[0], 'src/ccqp/ccqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    ccqpti = executable('ccqpti_'+val[0], 'src/ccqp/ccqpti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('ccqpt_'+val[0], ccqpt, suite: ['ccqp', 'fortran', val[0]], is_parallel : false)
    test('ccqpti_'+val[0], ccqpti, suite: ['ccqp', 'fortran', val[0]], is_parallel : false)

    qpat = executable('qpat_'+val[0], 'src/qpa/qpat.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    qpati = executable('qpati_'+val[0], 'src/qpa/qpati.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qpat_'+val[0], qpat, suite: ['qpa', 'fortran', val[0]], is_parallel : false)
    test('qpati_'+val[0], qpati, suite: ['qpa', 'fortran', val[0]], is_parallel : false)

    qpbt = executable('qpbt_'+val[0], 'src/qpb/qpbt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    qpbti = executable('qpbti_'+val[0], 'src/qpb/qpbti.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qpbt_'+val[0], qpbt, suite: ['qpb', 'fortran', val[0]], is_parallel : false)
    test('qpbti_'+val[0], qpbti, suite: ['qpb', 'fortran', val[0]], is_parallel : false)

    filtranet = executable('filtranet_'+val[0], 'src/filtrane/filtranet.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('filtranet_'+val[0], filtranet, suite: ['filtrane', 'fortran', val[0]], is_parallel : false)

    cdqpt = executable('cdqpt_'+val[0], 'src/cdqp/cdqpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('cdqpt_'+val[0], cdqpt, suite: ['cdqp', 'fortran', val[0]], is_parallel : false)

    qpt = executable('qpt_'+val[0], 'src/qp/qpt.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qpt_'+val[0], qpt, suite: ['qp', 'fortran', val[0]], is_parallel : false)

    qpct = executable('qpct_'+val[0], 'src/qpc/qpct.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
    test('qpct_'+val[0], qpct, suite: ['qpc', 'fortran', val[0]], is_parallel : false)

    if build_ssids
      ssidst = executable('ssidst_'+val[0], 'src/ssids/ssidst.F90', fortran_args : val[1], link_with : val[2], link_language: 'fortran', include_directories: libgalahad_include)
      test('ssidst_'+val[0], ssidst, suite: ['ssids', 'fortran', val[0]], is_parallel : false)
    endif
  endforeach
endif

if build_tests and build_ciface

  c_tests_folder = 'share/tests/C'
  c_tests = []
  if build_single
    c_tests += [['single', extra_args_single, libgalahad_c_single]]
  endif
  if build_double
    c_tests += [['double', extra_args_double, libgalahad_c_double]]
  endif

  foreach val: c_tests
    arct_c = executable('arct_c_'+val[0], 'src/arc/C/arct.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    arctf_c = executable('arctf_c_'+val[0], 'src/arc/C/arctf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('arct_c_'+val[0], arct_c, suite: ['arc', 'C', val[0]], is_parallel : false)
    test('arctf_c_'+val[0], arctf_c, suite: ['arc', 'C', val[0]], is_parallel : false)

    bgot_c = executable('bgot_c_'+val[0], 'src/bgo/C/bgot.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    bgotf_c = executable('bgotf_c_'+val[0], 'src/bgo/C/bgotf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('bgot_c_'+val[0], bgot_c, suite: ['bgo', 'C', val[0]], is_parallel : false)
    test('bgotf_c_'+val[0], bgotf_c, suite: ['bgo', 'C', val[0]], is_parallel : false)

    bllst_c = executable('bllst_c_'+val[0], 'src/blls/C/bllst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    bllstf_c = executable('bllstf_c_'+val[0], 'src/blls/C/bllstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('bllst_c_'+val[0], bllst_c, suite: ['blls', 'C', val[0]], is_parallel : false)
    test('bllstf_c_'+val[0], bllstf_c, suite: ['blls', 'C', val[0]], is_parallel : false)

    bqpt_c = executable('bqpt_c_'+val[0], 'src/bqp/C/bqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    bqptf_c = executable('bqptf_c_'+val[0], 'src/bqp/C/bqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('bqpt_c_'+val[0], bqpt_c, suite: ['bqp', 'C', val[0]], is_parallel : false)
    test('bqptf_c_'+val[0], bqptf_c, suite: ['bqp', 'C', val[0]], is_parallel : false)

    bqpbt_c = executable('bqpbt_c_'+val[0], 'src/bqpb/C/bqpbt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    bqpbtf_c = executable('bqpbtf_c_'+val[0], 'src/bqpb/C/bqpbtf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('bqpbt_c_'+val[0], bqpbt_c, suite: ['bqpb', 'C', val[0]], is_parallel : false)
    test('bqpbtf_c_'+val[0], bqpbtf_c, suite: ['bqpb', 'C', val[0]], is_parallel : false)

    ccqpt_c = executable('ccqpt_c_'+val[0], 'src/ccqp/C/ccqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    ccqptf_c = executable('ccqptf_c_'+val[0], 'src/ccqp/C/ccqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('ccqpt_c_'+val[0], ccqpt_c, suite: ['ccqp', 'C', val[0]], is_parallel : false)
    test('ccqptf_c_'+val[0], ccqptf_c, suite: ['ccqp', 'C', val[0]], is_parallel : false)

    cqpt_c = executable('cqpt_c_'+val[0], 'src/cqp/C/cqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    cqptf_c = executable('cqptf_c_'+val[0], 'src/cqp/C/cqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('cqpt_c_'+val[0], cqpt_c, suite: ['cqp', 'C', val[0]], is_parallel : false)
    test('cqptf_c_'+val[0], cqptf_c, suite: ['cqp', 'C', val[0]], is_parallel : false)

    dgot_c = executable('dgot_c_'+val[0], 'src/dgo/C/dgot.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    dgotf_c = executable('dgotf_c_'+val[0], 'src/dgo/C/dgotf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('dgot_c_'+val[0], dgot_c, suite: ['dgo', 'C', val[0]], is_parallel : false)
    test('dgotf_c_'+val[0], dgotf_c, suite: ['dgo', 'C', val[0]], is_parallel : false)

    dpst_c = executable('dpst_c_'+val[0], 'src/dps/C/dpst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    dpstf_c = executable('dpstf_c_'+val[0], 'src/dps/C/dpstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('dpst_c_'+val[0], dpst_c, suite: ['dps', 'C', val[0]], is_parallel : false)
    test('dpstf_c_'+val[0], dpstf_c, suite: ['dps', 'C', val[0]], is_parallel : false)

    dqpt_c = executable('dqpt_c_'+val[0], 'src/dqp/C/dqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    dqptf_c = executable('dqptf_c_'+val[0], 'src/dqp/C/dqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('dqpt_c_'+val[0], dqpt_c, suite: ['dqp', 'C', val[0]], is_parallel : false)
    test('dqptf_c_'+val[0], dqptf_c, suite: ['dqp', 'C', val[0]], is_parallel : false)

    eqpt_c = executable('eqpt_c_'+val[0], 'src/eqp/C/eqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    eqptf_c = executable('eqptf_c_'+val[0], 'src/eqp/C/eqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('eqpt_c_'+val[0], eqpt_c, suite: ['eqp', 'C', val[0]], is_parallel : false)
    test('eqptf_c_'+val[0], eqptf_c, suite: ['eqp', 'C', val[0]], is_parallel : false)

    fdct_c = executable('fdct_c_'+val[0], 'src/fdc/C/fdct.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    fdctf_c = executable('fdctf_c_'+val[0], 'src/fdc/C/fdctf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('fdct_c_'+val[0], fdct_c, suite: ['fdc', 'C', val[0]], is_parallel : false)
    test('fdctf_c_'+val[0], fdctf_c, suite: ['fdc', 'C', val[0]], is_parallel : false)

    glrtt_c = executable('glrtt_c_'+val[0], 'src/glrt/C/glrtt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('glrtt_c_'+val[0], glrtt_c, suite: ['glrt', 'C', val[0]], is_parallel : false)

    gltrt_c = executable('gltrt_c_'+val[0], 'src/gltr/C/gltrt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('gltrt_c_'+val[0], gltrt_c, suite: ['gltr', 'C', val[0]], is_parallel : false)

    l2rtt_c = executable('l2rtt_c_'+val[0], 'src/l2rt/C/l2rtt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('l2rtt_c_'+val[0], l2rtt_c, suite: ['l2rt', 'C', val[0]], is_parallel : false)

    lhst_c = executable('lhst_c_'+val[0], 'src/lhs/C/lhst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lhst_c_'+val[0], lhst_c, suite: ['lhs', 'C', val[0]], is_parallel : false)

    lpat_c = executable('lpat_c_'+val[0], 'src/lpa/C/lpat.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    lpatf_c = executable('lpatf_c_'+val[0], 'src/lpa/C/lpatf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lpat_c_'+val[0], lpat_c, suite: ['lpa', 'C', val[0]], is_parallel : false)
    test('lpatf_c_'+val[0], lpatf_c, suite: ['lpa', 'C', val[0]], is_parallel : false)

    lpbt_c = executable('lpbt_c_'+val[0], 'src/lpb/C/lpbt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    lpbtf_c = executable('lpbtf_c_'+val[0], 'src/lpb/C/lpbtf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lpbt_c_'+val[0], lpbt_c, suite: ['lpb', 'C', val[0]], is_parallel : false)
    test('lpbtf_c_'+val[0], lpbtf_c, suite: ['lpb', 'C', val[0]], is_parallel : false)

    lsqpt_c = executable('lsqpt_c_'+val[0], 'src/lsqp/C/lsqpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    lsqptf_c = executable('lsqptf_c_'+val[0], 'src/lsqp/C/lsqptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lsqpt_c_'+val[0], lsqpt_c, suite: ['lsqp', 'C', val[0]], is_parallel : false)
    test('lsqptf_c_'+val[0], lsqptf_c, suite: ['lsqp', 'C', val[0]], is_parallel : false)

    lsrtt_c = executable('lsrtt_c_'+val[0], 'src/lsrt/C/lsrtt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lsrtt_c_'+val[0], lsrtt_c, suite: ['lsrt', 'C', val[0]], is_parallel : false)

    lstrt_c = executable('lstrt_c_'+val[0], 'src/lstr/C/lstrt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('lstrt_c_'+val[0], lstrt_c, suite: ['lstr', 'C', val[0]], is_parallel : false)

    nlst_c = executable('nlst_c_'+val[0], 'src/nls/C/nlst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    nlstf_c = executable('nlstf_c_'+val[0], 'src/nls/C/nlstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('nlst_c_'+val[0], nlst_c, suite: ['nls', 'C', val[0]], is_parallel : false)
    test('nlstf_c_'+val[0], nlstf_c, suite: ['nls', 'C', val[0]], is_parallel : false)

    presolvet_c = executable('presolvet_c_'+val[0], 'src/presolve/C/presolvet.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    presolvetf_c = executable('presolvetf_c_'+val[0], 'src/presolve/C/presolvetf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('presolvet_c_'+val[0], presolvet_c, suite: ['presolve', 'C', val[0]], is_parallel : false)
    test('presolvetf_c_'+val[0], presolvetf_c, suite: ['presolve', 'C', val[0]], is_parallel : false)

    pslst_c = executable('pslst_c_'+val[0], 'src/psls/C/pslst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    pslstf_c = executable('pslstf_c_'+val[0], 'src/psls/C/pslstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('pslst_c_'+val[0], pslst_c, suite: ['psls', 'C', val[0]], is_parallel : false)
    test('pslstf_c_'+val[0], pslstf_c, suite: ['psls', 'C', val[0]], is_parallel : false)

    qpat_c = executable('qpat_c_'+val[0], 'src/qpa/C/qpat.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    qpatf_c = executable('qpatf_c_'+val[0], 'src/qpa/C/qpatf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('qpat_c_'+val[0], qpat_c, suite: ['qpa', 'C', val[0]], is_parallel : false)
    test('qpatf_c_'+val[0], qpatf_c, suite: ['qpa', 'C', val[0]], is_parallel : false)

    qpbt_c = executable('qpbt_c_'+val[0], 'src/qpb/C/qpbt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    qpbtf_c = executable('qpbtf_c_'+val[0], 'src/qpb/C/qpbtf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('qpbt_c_'+val[0], qpbt_c, suite: ['qpb', 'C', val[0]], is_parallel : false)
    test('qpbtf_c_'+val[0], qpbtf_c, suite: ['qpb', 'C', val[0]], is_parallel : false)

    rpdt_c = executable('rpdt_c_'+val[0], 'src/rpd/C/rpdt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    rpdtf_c = executable('rpdtf_c_'+val[0], 'src/rpd/C/rpdtf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('rpdt_c_'+val[0], rpdt_c, suite: ['rpd', 'C', val[0]], is_parallel : false)
    test('rpdtf_c_'+val[0], rpdtf_c, suite: ['rpd', 'C', val[0]], is_parallel : false)

    rqst_c = executable('rqst_c_'+val[0], 'src/rqs/C/rqst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    rqstf_c = executable('rqstf_c_'+val[0], 'src/rqs/C/rqstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('rqst_c_'+val[0], rqst_c, suite: ['rqs', 'C', val[0]], is_parallel : false)
    test('rqstf_c_'+val[0], rqstf_c, suite: ['rqs', 'C', val[0]], is_parallel : false)

    sblst_c = executable('sblst_c_'+val[0], 'src/sbls/C/sblst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    sblstf_c = executable('sblstf_c_'+val[0], 'src/sbls/C/sblstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('sblst_c_'+val[0], sblst_c, suite: ['sbls', 'C', val[0]], is_parallel : false)
    test('sblstf_c_'+val[0], sblstf_c, suite: ['sbls', 'C', val[0]], is_parallel : false)

    sllst_c = executable('sllst_c_'+val[0], 'src/slls/C/sllst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    sllstf_c = executable('sllstf_c_'+val[0], 'src/slls/C/sllstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('sllst_c_'+val[0], sllst_c, suite: ['slls', 'C', val[0]], is_parallel : false)
    test('sllstf_c_'+val[0], sllstf_c, suite: ['slls', 'C', val[0]], is_parallel : false)

    slst_c = executable('slst_c_'+val[0], 'src/sls/C/slst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    slstf_c = executable('slstf_c_'+val[0], 'src/sls/C/slstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('slst_c_'+val[0], slst_c, suite: ['sls', 'C', val[0]], is_parallel : false)
    test('slstf_c_'+val[0], slstf_c, suite: ['sls', 'C', val[0]], is_parallel : false)

    trbt_c = executable('trbt_c_'+val[0], 'src/trb/C/trbt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    trbtf_c = executable('trbtf_c_'+val[0], 'src/trb/C/trbtf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('trbt_c_'+val[0], trbt_c, suite: ['trb', 'C', val[0]], is_parallel : false)
    test('trbtf_c_'+val[0], trbtf_c, suite: ['trb', 'C', val[0]], is_parallel : false)

    trst_c = executable('trst_c_'+val[0], 'src/trs/C/trst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    trstf_c = executable('trstf_c_'+val[0], 'src/trs/C/trstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('trst_c_'+val[0], trst_c, suite: ['trs', 'C', val[0]], is_parallel : false)
    test('trstf_c_'+val[0], trstf_c, suite: ['trs', 'C', val[0]], is_parallel : false)

    trut_c = executable('trut_c_'+val[0], 'src/tru/C/trut.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    trutf_c = executable('trutf_c_'+val[0], 'src/tru/C/trutf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('trut_c_'+val[0], trut_c, suite: ['tru', 'C', val[0]], is_parallel : false)
    test('trutf_c_'+val[0], trutf_c, suite: ['tru', 'C', val[0]], is_parallel : false)

    ugot_c = executable('ugot_c_'+val[0], 'src/ugo/C/ugot.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('ugot_c_'+val[0], ugot_c, suite: ['ugo', 'C', val[0]], is_parallel : false)

    ulst_c = executable('ulst_c_'+val[0], 'src/uls/C/ulst.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    ulstf_c = executable('ulstf_c_'+val[0], 'src/uls/C/ulstf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('ulst_c_'+val[0], ulst_c, suite: ['uls', 'C', val[0]], is_parallel : false)
    test('ulstf_c_'+val[0], ulstf_c, suite: ['uls', 'C', val[0]], is_parallel : false)

    wcpt_c = executable('wcpt_c_'+val[0], 'src/wcp/C/wcpt.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    wcptf_c = executable('wcptf_c_'+val[0], 'src/wcp/C/wcptf.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include)
    test('wcpt_c_'+val[0], wcpt_c, suite: ['wcp', 'C', val[0]], is_parallel : false)
    test('wcptf_c_'+val[0], wcptf_c, suite: ['wcp', 'C', val[0]], is_parallel : false)
  endforeach
endif

# Example programs

if build_examples and build_double
  fortran_examples_folder = 'share/examples'
  executable('amds', 'src/amd/amds.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bscs', 'src/bsc/bscs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('checks', 'src/check/checks.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('converts', 'src/convert/converts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('fdhs', 'src/fdh/fdhs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('fdhs2', 'src/fdh/fdhs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('fits', 'src/fit/fits.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('gltrs', 'src/gltr/gltrs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('gltrs2', 'src/gltr/gltrs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('gltrs3', 'src/gltr/gltrs3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('glrts', 'src/glrt/glrts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('glss', 'src/gls/glss.f90', link_with : [libgalahad_hsl_double, libgalahad_double], link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('hashs', 'src/hash/hashs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lqrs', 'src/lqr/lqrs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lqrs2', 'src/lqr/lqrs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lqts', 'src/lqt/lqts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lqts2', 'src/lqt/lqts2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lsps', 'src/lsp/lsps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lstrs', 'src/lstr/lstrs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lstrs2', 'src/lstr/lstrs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lsrts', 'src/lsrt/lsrts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lsrts2', 'src/lsrt/lsrts2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('l2rts', 'src/l2rt/l2rts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('l2rts2', 'src/l2rt/l2rts2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lhss', 'src/lhs/lhss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lmss', 'src/lms/lmss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('miqrs', 'src/miqr/miqrs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('miqrs2', 'src/miqr/miqrs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('mops', 'src/mop/mops.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('nlpts', 'src/nlpt/nlpts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('presolves', 'src/presolve/presolves.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpps', 'src/qpp/qpps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpts', 'src/qpt/qpts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('rands', 'src/rand/rands.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('rpds', 'src/rpd/rpds.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('rootss', 'src/roots/rootss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('scales', 'src/scale/scales.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('scus', 'src/scu/scus.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('secs', 'src/sec/secs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('shas', 'src/sha/shas.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('silss', 'src/sils/silss.f90', link_with : [libgalahad_hsl_double, libgalahad_double], link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('smts', 'src/smt/smts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sorts', 'src/sort/sorts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ugos', 'src/ugo/ugos.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ugos2', 'src/ugo/ugos2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ugos3', 'src/ugo/ugos3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('slss', 'src/sls/slss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ulss', 'src/uls/ulss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sblss', 'src/sbls/sblss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('irs', 'src/ir/irs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('pslss', 'src/psls/pslss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trss', 'src/trs/trss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trss2', 'src/trs/trss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('icfss', 'src/icfs/icfss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss', 'src/slls/sllss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss2', 'src/slls/sllss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss3', 'src/slls/sllss3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss4', 'src/slls/sllss4.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss5', 'src/slls/sllss5.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss6', 'src/slls/sllss6.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('sllss7', 'src/slls/sllss7.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cros', 'src/cro/cros.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cros2', 'src/cro/cros2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cros3', 'src/cro/cros3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('fdcs', 'src/fdc/fdcs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('wcps', 'src/wcp/wcps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('rqss', 'src/rqs/rqss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('rqss2', 'src/rqs/rqss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bllss', 'src/blls/bllss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bllss2', 'src/blls/bllss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bllss3', 'src/blls/bllss3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bqps', 'src/bqp/bqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bqps2', 'src/bqp/bqps2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bqps3', 'src/bqp/bqps3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cqps', 'src/cqp/cqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cqps2', 'src/cqp/cqps2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cqps3', 'src/cqp/cqps3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dlps', 'src/dlp/dlps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dqps', 'src/dqp/dqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dqps2', 'src/dqp/dqps2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dqps3', 'src/dqp/dqps3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('eqps', 'src/eqp/eqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('l1qps', 'src/l1qp/l1qps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dpss', 'src/dps/dpss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dpss2', 'src/dps/dpss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lancelots', 'src/lancelot/lancelots.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lancelot_steerings', 'src/lancelot/lancelot_steerings.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder, include_directories: libgalahad_include)
  executable('lancelot_simples', 'src/lancelot_simple/lancelot_simples.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder, include_directories: libgalahad_include)
  executable('filtranes', 'src/filtrane/filtranes.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trbs', 'src/trb/trbs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trbs2', 'src/trb/trbs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trbs3', 'src/trb/trbs3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('nlss', 'src/nls/nlss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('nlss2', 'src/nls/nlss2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('nlss3', 'src/nls/nlss3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  if build_ssids
    executable('ssidss', 'src/ssids/ssidss.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  endif
  executable('trus', 'src/tru/trus.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trus2', 'src/tru/trus2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trus3', 'src/tru/trus3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('trus4', 'src/tru/trus4.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bgos', 'src/bgo/bgos.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bgos2', 'src/bgo/bgos2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('arcs', 'src/arc/arcs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('arcs2', 'src/arc/arcs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('arcs3', 'src/arc/arcs3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('arcs4', 'src/arc/arcs4.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dgos', 'src/dgo/dgos.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('dgos2', 'src/dgo/dgos2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bqpbs', 'src/bqpb/bqpbs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('bqpbs2', 'src/bqpb/bqpbs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ccqps', 'src/ccqp/ccqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ccqps2', 'src/ccqp/ccqps2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('ccqps3', 'src/ccqp/ccqps3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('cdqps', 'src/cdqp/cdqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('llsts', 'src/llst/llsts.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('llsts2', 'src/llst/llsts2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('llsts3', 'src/llst/llsts3.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lpas', 'src/lpa/lpas.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lpbs', 'src/lpb/lpbs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lsqps', 'src/lsqp/lsqps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('lsqps2', 'src/lsqp/lsqps2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpas', 'src/qpa/qpas.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpbs', 'src/qpb/qpbs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpbs2', 'src/qpb/qpbs2.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qpcs', 'src/qpc/qpcs.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  executable('qps', 'src/qp/qps.f90', link_with : libgalahad_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)

  # executable('ma48ss', 'src/external/ma48/ma48ss.f90', link_with : libgalahad_hsl_single, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  # executable('ma48ds', 'src/external/ma48/ma48ds.f90', link_with : libgalahad_hsl_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
endif

if build_examples and build_ciface

  c_examples_folder = 'share/examples/C'
  c_examples = []
  if build_single
    c_examples += [['single', extra_args_single, libgalahad_c_single]]
  endif
  if build_double
    c_examples += [['double', extra_args_double, libgalahad_c_double]]
  endif

  foreach val: c_examples
    arcs_c = executable('arcs_c_'+val[0], 'src/arc/C/arcs.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    arcs2_c = executable('arcs2_c_'+val[0], 'src/arc/C/arcs2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    arcs3_c = executable('arcs3_c_'+val[0], 'src/arc/C/arcs3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    arcs4_c = executable('arcs4_c_'+val[0], 'src/arc/C/arcs4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    bgos_c = executable('bgos_c_'+val[0], 'src/bgo/C/bgos.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    bgos2_c = executable('bgos2_c_'+val[0], 'src/bgo/C/bgos2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    bgos3_c = executable('bgos3_c_'+val[0], 'src/bgo/C/bgos3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    bgos4_c = executable('bgos4_c_'+val[0], 'src/bgo/C/bgos4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    dgos_c = executable('dgos_c_'+val[0], 'src/dgo/C/dgos.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    dgos2_c = executable('dgos2_c_'+val[0], 'src/dgo/C/dgos2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    dgos3_c = executable('dgos3_c_'+val[0], 'src/dgo/C/dgos3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    dgos4_c = executable('dgos4_c_'+val[0], 'src/dgo/C/dgos4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    lhs_c = executable('lhss_c_'+val[0], 'src/lhs/C/lhss.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    nlss_c = executable('nlss_c_'+val[0], 'src/nls/C/nlss.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    nlss2_c = executable('nlss2_c_'+val[0], 'src/nls/C/nlss2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    nlss3_c = executable('nlss3_c_'+val[0], 'src/nls/C/nlss3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    nlss4_c = executable('nlss4_c_'+val[0], 'src/nls/C/nlss4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trbs_c = executable('trbs_c_'+val[0], 'src/trb/C/trbs.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trbs2_c = executable('trbs2_c_'+val[0], 'src/trb/C/trbs2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trbs3_c = executable('trbs3_c_'+val[0], 'src/trb/C/trbs3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trbs4_c = executable('trbs4_c_'+val[0], 'src/trb/C/trbs4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trus_c = executable('trus_c_'+val[0], 'src/tru/C/trus.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trus2_c = executable('trus2_c_'+val[0], 'src/tru/C/trus2.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trus3_c = executable('trus3_c_'+val[0], 'src/tru/C/trus3.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    trus4_c = executable('trus4_c_'+val[0], 'src/tru/C/trus4.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    ugos_c = executable('ugos_c_'+val[0], 'src/ugo/C/ugos.c', c_args : val[1], link_with : val[2], link_language: 'c', link_args : '-lm', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
  endforeach
endif
