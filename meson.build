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
subdir('src/lapack')  # libgalahad_blas/lapack only contain the.F90 interfaces
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

# Decide what goes into libgalahad_hsl, if anything, based on information
# supplied by the user

libgalahad_hsl_single_src = []
libgalahad_hsl_double_src = []

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

libgalahad_hsl_single_src += files('src/kinds/kinds.F90')
libgalahad_hsl_double_src += files('src/kinds/kinds.F90')

libgalahad_hsl_single_src += files('src/symbols/symbols.F90')
libgalahad_hsl_double_src += files('src/symbols/symbols.F90')

# FIXME
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

libmkl_pardiso_src += files('src/dum/mkl_pardiso.F90', 'src/kinds/kinds.F90', 'src/symbols/symbols.F90')

libmkl_pardiso = library('mkl_pardiso',
                         sources : libmkl_pardiso_src,
                         link_with : [libgalahad_blas, libgalahad_lapack],
                         dependencies : blas_dep + lapack_dep,
                         install : true)

libpardiso_src += files('src/dum/pardiso.F90', 'src/kinds/kinds.F90', 'src/symbols/symbols.F90')

libpardiso = library('pardiso',
                     sources : libpardiso_src,
                     link_with : [libgalahad_blas, libgalahad_lapack],
                     dependencies : blas_dep + lapack_dep,
                     include_directories: libgalahad_include,
                     install : true)

libspral_ssids_single_src += files('src/dum/ssidss.F90', 'src/kinds/kinds.F90', 'src/symbols/symbols.F90')
libspral_ssids_double_src += files('src/dum/ssidsd.F90', 'src/kinds/kinds.F90', 'src/symbols/symbols.F90')

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

libmumps_src += files('src/dum/mumps.F90', 'src/dum/mpi.F90', 'src/kinds/kinds.F90')

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

libspmf_src += files('src/dum/spmf_enums.F90', 'src/dum/spmf_interfaces.F90', 'src/dum/spmf.F90', 'src/kinds/kinds.F90')

libspmf = library('spmf',
                  sources : libspmf_src,
                  link_with : [libgalahad_blas, libgalahad_lapack],
                  dependencies : blas_dep + lapack_dep,
                  install : true)

libpastix_src += files('src/dum/pastixf_enums.F90', 'src/dum/pastixf_interfaces.F90', 'src/dum/pastixf.F90', 'src/kinds/kinds.F90')

libpastix_single = library('pastix_single',
                    sources : libpastix_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libspmf],
                    dependencies : blas_dep + lapack_dep,
                    fortran_args : extra_args_single,
                    install : true)

libpastix_double = library('pastix_double',
                    sources : libpastix_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libspmf],
                    dependencies : blas_dep + lapack_dep,
                    fortran_args : extra_args_double,
                    install : true)

libwsmp_src += files('src/dum/wsmp.F90', 'src/kinds/kinds.F90', 'src/symbols/symbols.F90')

libwsmp = library('wsmp',
                  sources : libwsmp_src,
                  link_with : [libgalahad_blas, libgalahad_lapack],
                  dependencies : blas_dep + lapack_dep,
                  include_directories: libgalahad_include,
                  install : true)

libgalahad_metis = library('galahad_metis',
                           sources : 'src/dum/metis4.f',
                           install : true)

## 1. the following modules have no dependencies inside libgalahad; they appear in alphabetical order
subdir('src/clock')
subdir('src/checkpoint')
subdir('src/common')
subdir('src/dum')
subdir('src/copyright')
subdir('src/extend')
subdir('src/kinds')
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

## 2. the following modules have dependencies
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

# needs CUTEst interface
# subdir('src/ptrans')
# subdir('src/cutest_functions')

subdir('src/lancelot')
subdir('src/lancelot_simple')
subdir('src/filtrane')
subdir('src/psls')
subdir('src/trb')
subdir('src/tru')
subdir('src/bgo')
subdir('src/nls')
subdir('src/arc')
subdir('src/dgo')
subdir('src/bqpb')
subdir('src/ccqp')
subdir('src/cdqp')

# Issue 64
# subdir('src/clls')

# needs c++ files
# subdir('src/ssids')

libgalahad_single = library('galahad_single',
                    sources : libgalahad_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libgalahad_hsl_single,
                                 libpardiso, libmkl_pardiso, libspral_ssids_single, libmumps_single,
                                 libspmf, libwsmp, libpastix_single, libgalahad_metis],
                    dependencies : libgalahad_deps,
                    fortran_args : extra_args_single,
                    include_directories: libgalahad_include,
                    install : true)

libgalahad_double = library('galahad_double',
                    sources : libgalahad_src,
                    link_with : [libgalahad_blas, libgalahad_lapack, libgalahad_hsl_double,
                                 libpardiso, libmkl_pardiso, libspral_ssids_double, libmumps_double,
                                 libspmf, libwsmp, libpastix_double, libgalahad_metis],
                    dependencies : libgalahad_deps,
                    fortran_args : extra_args_double,
                    include_directories: libgalahad_include,
                    install : true)

build_ciface = get_option('ciface')
build_pythoniface = get_option('pythoniface')

if build_ciface
  libgalahad_c_single = library('galahad_c_single',
                        sources : dum_single_c_src + libgalahad_c_src,
                        link_with : [libgalahad_single, libgalahad_hsl_single, libspral_ssids_single],
                        fortran_args : extra_args_single,
                        include_directories: libgalahad_include,
                        install : true)

  libgalahad_c_double = library('galahad_c_double',
                        sources : dum_double_c_src + libgalahad_c_src,
                        link_with : [libgalahad_double, libgalahad_hsl_double, libspral_ssids_double],
                        fortran_args : extra_args_double,
                        include_directories: libgalahad_include,
                        install : true)
endif

# Executables

buildspec_single = executable('buildspec_single', 'src/buildspec/buildspec.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include, install : true)
buildspec_double = executable('buildspec_double', 'src/buildspec/buildspec.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include, install : true)

# Unit tests

amdt_single = executable('amdt_single', 'src/amd/amdt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
amdt_double = executable('admt_double', 'src/amd/amdt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('amd_single', amdt_single, suite: ['fortran', 'single'], is_parallel : false)
test('amd_double', amdt_double, suite: ['fortran', 'double'], is_parallel : false)

bsct_single = executable('bsct_single', 'src/bsc/bsct.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bsct_double = executable('bsct_double', 'src/bsc/bsct.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('bsc_single', bsct_single, suite: ['fortran', 'single'], is_parallel : false)
test('bsc_double', bsct_double, suite: ['fortran', 'double'], is_parallel : false)

checkt_single = executable('checkt_single', 'src/check/checkt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
checkt_double = executable('checkt_double', 'src/check/checkt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('check_single', checkt_single, suite: ['fortran', 'single'], is_parallel : false)
test('check_double', checkt_double, suite: ['fortran', 'double'], is_parallel : false)

convertt_single = executable('convertt_single', 'src/convert/convertt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
convertt_double = executable('convertt_double', 'src/convert/convertt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('convert_single', convertt_single, suite: ['fortran', 'single'], is_parallel : false)
test('convert_double', convertt_double, suite: ['fortran', 'double'], is_parallel : false)

fdht_single = executable('fdht_single', 'src/fdh/fdht.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
fdht_double = executable('fdht_double', 'src/fdh/fdht.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('fdh_single', fdht_single, suite: ['fortran', 'single'], is_parallel : false)
test('fdh_double', fdht_double, suite: ['fortran', 'double'], is_parallel : false)

fitt_single = executable('fitt_single', 'src/fit/fitt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
fitt_double = executable('fitt_double', 'src/fit/fitt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('fit_single', fitt_single, suite: ['fortran', 'single'], is_parallel : false)
test('fit_double', fitt_double, suite: ['fortran', 'double'], is_parallel : false)

gltrt_single = executable('gltrt_single', 'src/gltr/gltrt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
gltrt_double = executable('gltrt_double', 'src/gltr/gltrt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
gltrti_single = executable('gltrti_single', 'src/gltr/gltrti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
gltrti_double = executable('gltrti_double', 'src/gltr/gltrti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('gltr1_single', gltrt_single, suite: ['fortran', 'single'], is_parallel : false)
test('gltr1_double', gltrt_double, suite: ['fortran', 'double'], is_parallel : false)
test('gltr2_single', gltrti_single, suite: ['fortran', 'single'], is_parallel : false)
test('gltr2_double', gltrti_double, suite: ['fortran', 'double'], is_parallel : false)

dgot_single = executable('dgot_single', 'src/dgo/dgot.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
dgot_double = executable('dgot_double', 'src/dgo/dgot.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
dgoti_single = executable('dgoti_single', 'src/dgo/dgoti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
dgoti_double = executable('dgoti_double', 'src/dgo/dgoti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('dgo1_single', dgot_single, suite: ['fortran', 'single'], is_parallel : false)
test('dgo1_double', dgot_double, suite: ['fortran', 'double'], is_parallel : false)
test('dgo2_single', dgoti_single, suite: ['fortran', 'single'], is_parallel : false)
test('dgo2_double', dgoti_double, suite: ['fortran', 'double'], is_parallel : false)

glrtt_single = executable('glrtt_single', 'src/glrt/glrtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
glrtt_double = executable('glrtt_double', 'src/glrt/glrtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('glrt_single', glrtt_single, suite: ['fortran', 'single'], is_parallel : false)
test('glrt_double', glrtt_double, suite: ['fortran', 'double'], is_parallel : false)

glst_single = executable('glst_single', 'src/gls/glst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
glst_double = executable('glst_double', 'src/gls/glst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('gls_single', glst_single, suite: ['fortran', 'single'], is_parallel : false)
test('gls_double', glst_double, suite: ['fortran', 'double'], is_parallel : false)

crot_single = executable('crot_single', 'src/cro/crot.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
crot_double = executable('crot_double', 'src/cro/crot.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('cro_single', crot_single, suite: ['fortran', 'single'], is_parallel : false)
test('cro_double', crot_double, suite: ['fortran', 'double'], is_parallel : false)

hasht_single = executable('hasht_single', 'src/hash/hasht.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
hasht_double = executable('hasht_double', 'src/hash/hasht.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('hash_single', hasht_single, suite: ['fortran', 'single'], is_parallel : false)
test('hash_double', hasht_double, suite: ['fortran', 'double'], is_parallel : false)

lqrt_single = executable('lqrt_single', 'src/lqr/lqrt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lqrt_double = executable('lqrt_double', 'src/lqr/lqrt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lqr_single', lqrt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lqr_double', lqrt_double, suite: ['fortran', 'double'], is_parallel : false)

lancelot_simplet_single = executable('lancelot_simplet_single', 'src/lancelot_simple/lancelot_simplet.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lancelot_simplet_double = executable('lancelot_simplet_double', 'src/lancelot_simple/lancelot_simplet.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lancelot_simple_single', lancelot_simplet_single, suite: ['fortran', 'single'], is_parallel : false)
test('lancelot_simple_double', lancelot_simplet_double, suite: ['fortran', 'double'], is_parallel : false)

lqtt_single = executable('lqtt_single', 'src/lqt/lqtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lqtt_double = executable('lqtt_double', 'src/lqt/lqtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
lqtt2_single = executable('lqtt2_single', 'src/lqt/lqtt2.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lqtt2_double = executable('lqtt2_double', 'src/lqt/lqtt2.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lqt_single', lqtt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lqt_double', lqtt_double, suite: ['fortran', 'double'], is_parallel : false)
test('lqt2_single', lqtt2_single, suite: ['fortran', 'single'], is_parallel : false)
test('lqt2_double', lqtt2_double, suite: ['fortran', 'double'], is_parallel : false)

lspt_single = executable('lspt_single', 'src/lsp/lspt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lspt_double = executable('lspt_double', 'src/lsp/lspt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lsp_single', lspt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lsp_double', lspt_double, suite: ['fortran', 'double'], is_parallel : false)

lstrt_single = executable('lstrt_single', 'src/lstr/lstrt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lstrt_double = executable('lstrt_double', 'src/lstr/lstrt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
lstrti_single = executable('lstrti_single', 'src/lstr/lstrti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lstrti_double = executable('lstrti_double', 'src/lstr/lstrti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lstr1_single', lstrt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lstr1_double', lstrt_double, suite: ['fortran', 'double'], is_parallel : false)
test('lstr2_single', lstrti_single, suite: ['fortran', 'single'], is_parallel : false)
test('lstr2_double', lstrti_double, suite: ['fortran', 'double'], is_parallel : false)

fdct_single = executable('fdct_single', 'src/fdc/fdct.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
fdct_double = executable('fdct_double', 'src/fdc/fdct.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
fdcti_single = executable('fdcti_single', 'src/fdc/fdcti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
fdcti_double = executable('fdcti_double', 'src/fdc/fdcti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('fdc1_single', fdct_single, suite: ['fortran', 'single'], is_parallel : false)
test('fdc1_double', fdct_double, suite: ['fortran', 'double'], is_parallel : false)
test('fdc2_single', fdcti_single, suite: ['fortran', 'single'], is_parallel : false)
test('fdc2_double', fdcti_double, suite: ['fortran', 'double'], is_parallel : false)

lsrtt_single = executable('lsrtt_single', 'src/lsrt/lsrtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lsrtt_double = executable('lsrtt_double', 'src/lsrt/lsrtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
lsrtti_single = executable('lsrtti_single', 'src/lsrt/lsrtti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lsrtti_double = executable('lsrtti_double', 'src/lsrt/lsrtti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lsrt1_single', lsrtt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lsrt1_double', lsrtt_double, suite: ['fortran', 'double'], is_parallel : false)
test('lsrt2_single', lsrtti_single, suite: ['fortran', 'single'], is_parallel : false)
test('lsrt2_double', lsrtti_double, suite: ['fortran', 'double'], is_parallel : false)

l2rtt_single  = executable('l2rtt_single' , 'src/l2rt/l2rtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
l2rtt_double  = executable('l2rtt_double' , 'src/l2rt/l2rtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
l2rtti_single = executable('l2rtti_single', 'src/l2rt/l2rtti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
l2rtti_double = executable('l2rtti_double', 'src/l2rt/l2rtti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('l2rt1_single', l2rtt_single , suite: ['fortran', 'single'], is_parallel : false)
test('l2rt1_double', l2rtt_double , suite: ['fortran', 'double'], is_parallel : false)
test('l2rt2_single', l2rtti_single, suite: ['fortran', 'single'], is_parallel : false)
test('l2rt2_double', l2rtti_double, suite: ['fortran', 'double'], is_parallel : false)

rqst_single  = executable('rqst_single' , 'src/rqs/rqst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rqst_double  = executable('rqst_double' , 'src/rqs/rqst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
rqsti_single = executable('rqsti_single', 'src/rqs/rqsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rqsti_double = executable('rqsti_double', 'src/rqs/rqsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('rqs1_single', rqst_single , suite: ['fortran', 'single'], is_parallel : false)
test('rqs1_double', rqst_double , suite: ['fortran', 'double'], is_parallel : false)
test('rqs2_single', rqsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('rqs2_double', rqsti_double, suite: ['fortran', 'double'], is_parallel : false)

lhst_single = executable('lhst_single', 'src/lhs/lhst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lhst_double = executable('lhst_double', 'src/lhs/lhst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lhs_single', lhst_single, suite: ['fortran', 'single'], is_parallel : false)
test('lhs_double', lhst_double, suite: ['fortran', 'double'], is_parallel : false)

lmst_single = executable('lmst_single', 'src/lms/lmst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lmst_double = executable('lmst_double', 'src/lms/lmst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lms_single', lmst_single, suite: ['fortran', 'single'], is_parallel : false)
test('lms_double', lmst_double, suite: ['fortran', 'double'], is_parallel : false)

miqrt_single = executable('miqrt_single', 'src/miqr/miqrt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
miqrt_double = executable('miqrt_double', 'src/miqr/miqrt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('miqr_single', miqrt_single, suite: ['fortran', 'single'], is_parallel : false)
test('miqr_double', miqrt_double, suite: ['fortran', 'double'], is_parallel : false)

mopt_single = executable('mopt_single', 'src/mop/mopt.F90', fortran_args : extra_args_single, link_with : [libgalahad_single, libgalahad_blas], include_directories: libgalahad_include)
mopt_double = executable('mopt_double', 'src/mop/mopt.F90', fortran_args : extra_args_double, link_with : [libgalahad_double, libgalahad_blas], include_directories: libgalahad_include)
test('mop_single', mopt_single, suite: ['fortran', 'single'], is_parallel : false)
test('mop_double', mopt_double, suite: ['fortran', 'double'], is_parallel : false)

dpst_single = executable('dpst_single', 'src/dps/dpst.F90', fortran_args : extra_args_single, link_with : [libgalahad_single, libgalahad_blas], include_directories: libgalahad_include)
dpst_double = executable('dpst_double', 'src/dps/dpst.F90', fortran_args : extra_args_double, link_with : [libgalahad_double, libgalahad_blas], include_directories: libgalahad_include)
test('dps_single', dpst_single, suite: ['fortran', 'single'], is_parallel : false)
test('dps_double', dpst_double, suite: ['fortran', 'double'], is_parallel : false)

sllst_single = executable('sllst_single', 'src/slls/sllst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sllst_double = executable('sllst_double', 'src/slls/sllst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
sllsti_single = executable('sllsti_single', 'src/slls/sllsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sllsti_double = executable('sllsti_double', 'src/slls/sllsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('slls1_single', sllst_single, suite: ['fortran', 'single'], is_parallel : false)
test('slls1_double', sllst_double, suite: ['fortran', 'double'], is_parallel : false)
test('slls2_single', sllsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('slls2_double', sllsti_double, suite: ['fortran', 'double'], is_parallel : false)

nlptt_single = executable('nlptt_single', 'src/nlpt/nlptt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
nlptt_double = executable('nlptt_double', 'src/nlpt/nlptt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('nlpt_single', nlptt_single, suite: ['fortran', 'single'], is_parallel : false)
test('nlpt_double', nlptt_double, suite: ['fortran', 'double'], is_parallel : false)

presolvet_single = executable('presolvet_single', 'src/presolve/presolvet.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
presolvet_double = executable('presolvet_double', 'src/presolve/presolvet.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
presolveti_single = executable('presolveti_single', 'src/presolve/presolveti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
presolveti_double = executable('presolveti_double', 'src/presolve/presolveti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('presolve1_single', presolvet_single, suite: ['fortran', 'single'], is_parallel : false)
test('presolve1_double', presolvet_double, suite: ['fortran', 'double'], is_parallel : false)
test('presolve2_single', presolveti_single, suite: ['fortran', 'single'], is_parallel : false)
test('presolve2_double', presolveti_double, suite: ['fortran', 'double'], is_parallel : false)

qppt_single = executable('qppt_single', 'src/qpp/qppt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
qppt_double = executable('qppt_double', 'src/qpp/qppt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('qpp_single', qppt_single, suite: ['fortran', 'single'], is_parallel : false)
test('qpp_double', qppt_double, suite: ['fortran', 'double'], is_parallel : false)

qptt_single = executable('qptt_single', 'src/qpt/qptt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
qptt_double = executable('qptt_double', 'src/qpt/qptt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('qpt_single', qptt_single, suite: ['fortran', 'single'], is_parallel : false)
test('qpt_double', qptt_double, suite: ['fortran', 'double'], is_parallel : false)

randt_single = executable('randt_single', 'src/rand/randt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
randt_double = executable('randt_double', 'src/rand/randt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('rand_single', randt_single, suite: ['fortran', 'single'], is_parallel : false)
test('rand_double', randt_double, suite: ['fortran', 'double'], is_parallel : false)

rpdt_single = executable('rpdt_single', 'src/rpd/rpdt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rpdt_double = executable('rpdt_double', 'src/rpd/rpdt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
rpdti_single = executable('rpdti_single', 'src/rpd/rpdti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rpdti_double = executable('rpdti_double', 'src/rpd/rpdti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('rpd1_single', rpdt_single, suite: ['fortran', 'single'], is_parallel : false)
test('rpd1_double', rpdt_double, suite: ['fortran', 'double'], is_parallel : false)
test('rpd2_single', rpdti_single, suite: ['fortran', 'single'], is_parallel : false)
test('rpd2_double', rpdti_double, suite: ['fortran', 'double'], is_parallel : false)

bllst_single = executable('bllst_single', 'src/blls/bllst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bllst_double = executable('bllst_double', 'src/blls/bllst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
bllsti_single = executable('bllsti_single', 'src/blls/bllsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bllsti_double = executable('bllsti_double', 'src/blls/bllsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('blls1_single', bllst_single, suite: ['fortran', 'single'], is_parallel : false)
test('blls1_double', bllst_double, suite: ['fortran', 'double'], is_parallel : false)
test('blls2_single', bllsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('blls2_double', bllsti_double, suite: ['fortran', 'double'], is_parallel : false)

trut_single = executable('trut_single', 'src/tru/trut.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
trut_double = executable('trut_double', 'src/tru/trut.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
truti_single = executable('truti_single', 'src/tru/truti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
truti_double = executable('truti_double', 'src/tru/truti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('tru1_single', trut_single, suite: ['fortran', 'single'], is_parallel : false)
test('tru1_double', trut_double, suite: ['fortran', 'double'], is_parallel : false)
test('tru2_single', truti_single, suite: ['fortran', 'single'], is_parallel : false)
test('tru2_double', truti_double, suite: ['fortran', 'double'], is_parallel : false)

rootst_single = executable('rootst_single', 'src/roots/rootst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rootst_double = executable('rootst_double', 'src/roots/rootst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('roots_single', rootst_single, suite: ['fortran', 'single'], is_parallel : false)
test('roots_double', rootst_double, suite: ['fortran', 'double'], is_parallel : false)

scut_single = executable('scut_single', 'src/scu/scut.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
scut_double = executable('scut_double', 'src/scu/scut.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('scu_single', scut_single, suite: ['fortran', 'single'], is_parallel : false)
test('scu_double', scut_double, suite: ['fortran', 'double'], is_parallel : false)

scalet_single = executable('scalet_single', 'src/scale/scalet.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
scalet_double = executable('scalet_double', 'src/scale/scalet.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('scale1_single', scalet_single, suite: ['fortran', 'single'], is_parallel : false)
test('scale1_double', scalet_double, suite: ['fortran', 'double'], is_parallel : false)

sect_single = executable('sect_single', 'src/sec/sect.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sect_double = executable('sect_double', 'src/sec/sect.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sec_single', sect_single, suite: ['fortran', 'single'], is_parallel : false)
test('sec_double', sect_double, suite: ['fortran', 'double'], is_parallel : false)

shat_single = executable('shat_single', 'src/sha/shat.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
shat_double = executable('shat_double', 'src/sha/shat.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sha_single', shat_single, suite: ['fortran', 'single'], is_parallel : false)
test('sha_double', shat_double, suite: ['fortran', 'double'], is_parallel : false)

silst_single = executable('silst_single', 'src/sils/silst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
silst_double = executable('silst_double', 'src/sils/silst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sils_single', silst_single, suite: ['fortran', 'single'], is_parallel : false)
test('sils_double', silst_double, suite: ['fortran', 'double'], is_parallel : false)

smtt_single = executable('smtt_single', 'src/smt/smtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
smtt_double = executable('smtt_double', 'src/smt/smtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('smt_single', smtt_single, suite: ['fortran', 'single'], is_parallel : false)
test('smt_double', smtt_double, suite: ['fortran', 'double'], is_parallel : false)

sortt_single = executable('sortt_single', 'src/sort/sortt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sortt_double = executable('sortt_double', 'src/sort/sortt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sort_single', sortt_single, suite: ['fortran', 'single'], is_parallel : false)
test('sort_double', sortt_double, suite: ['fortran', 'double'], is_parallel : false)

ugot_single = executable('ugot_single', 'src/ugo/ugot.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ugot_double = executable('ugot_double', 'src/ugo/ugot.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# ugot2_single = executable('ugot2_single', 'src/ugo/ugot2.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
# ugot2_double = executable('ugot2_double', 'src/ugo/ugot2.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('ugo1_single', ugot_single, suite: ['fortran', 'single'], is_parallel : false)
test('ugo1_double', ugot_double, suite: ['fortran', 'double'], is_parallel : false)
# test('ugo2_single', ugot2_single, suite: ['fortran', 'single'], is_parallel : false)
# test('ugo2_double', ugot2_double, suite: ['fortran', 'double'], is_parallel : false)

mkl_pardisot_single = executable('mkl_pardisot_single', 'src/external/mkl/mkl_pardisot.F90', fortran_args : extra_args_single, link_with : libmkl_pardiso, include_directories: libgalahad_include)
mkl_pardisot_double = executable('mkl_pardisot_double', 'src/external/mkl/mkl_pardisot.F90', fortran_args : extra_args_double, link_with : libmkl_pardiso, include_directories: libgalahad_include)
test('mkl_pardiso_single', mkl_pardisot_single, suite: ['fortran', 'single'], is_parallel : false)
test('mkl_pardiso_double', mkl_pardisot_double, suite: ['fortran', 'double'], is_parallel : false)

mumpst_single = executable('mumpst_single', 'src/external/mumps/mumpst.F90', fortran_args : extra_args_single, link_with : libmumps_single, include_directories: libgalahad_include)
mumpst_double = executable('mumpst_double', 'src/external/mumps/mumpst.F90', fortran_args : extra_args_double, link_with : libmumps_double, include_directories: libgalahad_include)
test('mumps_single', mumpst_single, suite: ['fortran', 'single'], is_parallel : false)
test('mumps_double', mumpst_double, suite: ['fortran', 'double'], is_parallel : false)

pardisot_single = executable('pardisot_single', 'src/external/pardiso/pardisot.F90', fortran_args : extra_args_single, link_with : libpardiso, include_directories: libgalahad_include)
pardisot_double = executable('pardisot_double', 'src/external/pardiso/pardisot.F90', fortran_args : extra_args_double, link_with : libpardiso, include_directories: libgalahad_include)
test('pardiso_single', pardisot_single, suite: ['fortran', 'single'], is_parallel : false)
test('pardiso_double', pardisot_double, suite: ['fortran', 'double'], is_parallel : false)

slst_single = executable('slst_single', ['src/sls/slst.F90', 'src/dum/mpi.F90'], fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
slst_double = executable('slst_double', ['src/sls/slst.F90', 'src/dum/mpi.F90'], fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
slsti_single = executable('slsti_single', 'src/sls/slsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
slsti_double = executable('slsti_double', 'src/sls/slsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sls1_single', slst_single, suite: ['fortran', 'single'], is_parallel : false)
test('sls1_double', slst_double, suite: ['fortran', 'double'], is_parallel : false)
test('sls2_single', slsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('sls2_double', slsti_double, suite: ['fortran', 'double'], is_parallel : false)

ulst_single = executable('ulst_single', 'src/uls/ulst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ulst_double = executable('ulst_double', 'src/uls/ulst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
ulsti_single = executable('ulsti_single', 'src/uls/ulsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ulsti_double = executable('ulsti_double', 'src/uls/ulsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('uls1_single', ulst_single, suite: ['fortran', 'single'], is_parallel : false)
test('uls1_double', ulst_double, suite: ['fortran', 'double'], is_parallel : false)
test('uls2_single', ulsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('uls2_double', ulsti_double, suite: ['fortran', 'double'], is_parallel : false)

irt_single = executable('irt_single', 'src/ir/irt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
irt_double = executable('irt_double', 'src/ir/irt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('ir_single', irt_single, suite: ['fortran', 'single'], is_parallel : false)
test('ir_double', irt_double, suite: ['fortran', 'double'], is_parallel : false)

icfst_single = executable('icfst_single', 'src/icfs/icfst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
icfst_double = executable('icfst_double', 'src/icfs/icfst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('icfs_single', icfst_single, suite: ['fortran', 'single'], is_parallel : false)
test('icfs_double', icfst_double, suite: ['fortran', 'double'], is_parallel : false)

trst_single = executable('trst_single', 'src/trs/trst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
trst_double = executable('trst_double', 'src/trs/trst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
trsti_single = executable('trsti_single', 'src/trs/trsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
trsti_double = executable('trsti_double', 'src/trs/trsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('trs1_single', trst_single, suite: ['fortran', 'single'], is_parallel : false)
test('trs1_double', trst_double, suite: ['fortran', 'double'], is_parallel : false)
test('trs2_single', trsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('trs2_double', trsti_double, suite: ['fortran', 'double'], is_parallel : false)

trbt_single = executable('trbt_single', 'src/trb/trbt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
trbt_double = executable('trbt_double', 'src/trb/trbt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
trbti_single = executable('trbti_single', 'src/trb/trbti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
trbti_double = executable('trbti_double', 'src/trb/trbti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('trb1_single', trbt_single, suite: ['fortran', 'single'], is_parallel : false)
test('trb1_double', trbt_double, suite: ['fortran', 'double'], is_parallel : false)
test('trb2_single', trbti_single, suite: ['fortran', 'single'], is_parallel : false)
test('trb2_double', trbti_double, suite: ['fortran', 'double'], is_parallel : false)

wcpt_single = executable('wcpt_single', 'src/wcp/wcpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
wcpt_double = executable('wcpt_double', 'src/wcp/wcpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
wcpti_single = executable('wcpti_single', 'src/wcp/wcpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
wcpti_double = executable('wcpti_double', 'src/wcp/wcpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('wcp1_single', wcpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('wcp1_double', wcpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('wcp2_single', wcpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('wcp2_double', wcpti_double, suite: ['fortran', 'double'], is_parallel : false)

pslst_single = executable('pslst_single', 'src/psls/pslst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
pslst_double = executable('pslst_double', 'src/psls/pslst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
pslsti_single = executable('pslsti_single', 'src/psls/pslsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
pslsti_double = executable('pslsti_double', 'src/psls/pslsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('psls1_single', pslst_single, suite: ['fortran', 'single'], is_parallel : false)
test('psls1_double', pslst_double, suite: ['fortran', 'double'], is_parallel : false)
test('psls2_single', pslsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('psls2_double', pslsti_double, suite: ['fortran', 'double'], is_parallel : false)

arct_single = executable('arct_single', 'src/arc/arct.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
arct_double = executable('arct_double', 'src/arc/arct.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
arcti_single = executable('arcti_single', 'src/arc/arcti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
arcti_double = executable('arcti_double', 'src/arc/arcti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('arc1_single', arct_single, suite: ['fortran', 'single'], is_parallel : false)
test('arc1_double', arct_double, suite: ['fortran', 'double'], is_parallel : false)
test('arc2_single', arcti_single, suite: ['fortran', 'single'], is_parallel : false)
test('arc2_double', arcti_double, suite: ['fortran', 'double'], is_parallel : false)

sblst_single = executable('sblst_single', 'src/sbls/sblst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sblst_double = executable('sblst_double', 'src/sbls/sblst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
sblsti_single = executable('sblsti_single', 'src/sbls/sblsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
sblsti_double = executable('sblsti_double', 'src/sbls/sblsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('sbls1_single', sblst_single, suite: ['fortran', 'single'], is_parallel : false)
test('sbls1_double', sblst_double, suite: ['fortran', 'double'], is_parallel : false)
test('sbls2_single', sblsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('sbls2_double', sblsti_double, suite: ['fortran', 'double'], is_parallel : false)

cqpt_single = executable('cqpt_single', 'src/cqp/cqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
cqpt_double = executable('cqpt_double', 'src/cqp/cqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
cqpti_single = executable('cqpti_single', 'src/cqp/cqpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
cqpti_double = executable('cqpti_double', 'src/cqp/cqpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('cqp1_single', cqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('cqp1_double', cqpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('cqp2_single', cqpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('cqp2_double', cqpti_double, suite: ['fortran', 'double'], is_parallel : false)

bqpt_single = executable('bqpt_single', 'src/bqp/bqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bqpt_double = executable('bqpt_double', 'src/bqp/bqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
bqpti_single = executable('bqpti_single', 'src/bqp/bqpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bqpti_double = executable('bqpti_double', 'src/bqp/bqpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('bqp1_single', bqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('bqp1_double', bqpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('bqp2_single', bqpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('bqp2_double', bqpti_double, suite: ['fortran', 'double'], is_parallel : false)

wsmpt_single = executable('wsmpt_single', 'src/external/wsmp/wsmpt.F90', fortran_args : extra_args_single, link_with : libwsmp, include_directories: libgalahad_include)
wsmpt_double = executable('wsmpt_double', 'src/external/wsmp/wsmpt.F90', fortran_args : extra_args_double, link_with : libwsmp, include_directories: libgalahad_include)
test('wsmp_single', wsmpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('wsmp_double', wsmpt_double, suite: ['fortran', 'double'], is_parallel : false)

lancelott_single = executable('lancelott_single', 'src/lancelot/lancelott.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lancelott_double = executable('lancelott_double', 'src/lancelot/lancelott.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lancelot_single', lancelott_single, suite: ['fortran', 'single'], is_parallel : false)
test('lancelot_double', lancelott_double, suite: ['fortran', 'double'], is_parallel : false)

lancelot_steeringt_single = executable('lancelot_steeringt_single', 'src/lancelot/lancelot_steeringt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lancelot_steeringt_double = executable('lancelot_steeringt_double', 'src/lancelot/lancelot_steeringt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lancelot_steering_single', lancelot_steeringt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lancelot_steering_double', lancelot_steeringt_double, suite: ['fortran', 'double'], is_parallel : false)

# Issue 56
# l1qpt_single = executable('l1qpt_single', 'src/l1qp/l1qpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
l1qpt_double = executable('l1qpt_double', 'src/l1qp/l1qpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# test('l1qp_single', l1qpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('l1qp_double', l1qpt_double, suite: ['fortran', 'double'], is_parallel : false)

dlpt_single = executable('dlpt_single', 'src/dlp/dlpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
dlpt_double = executable('dlpt_double', 'src/dlp/dlpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('dlp_single', dlpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('dlp_double', dlpt_double, suite: ['fortran', 'double'], is_parallel : false)

dqpt_single = executable('dqpt_single', 'src/dqp/dqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
dqpt_double = executable('dqpt_double', 'src/dqp/dqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
dqpti_single = executable('dqpti_single', 'src/dqp/dqpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
dqpti_double = executable('dqpti_double', 'src/dqp/dqpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('dqp1_single', dqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('dqp1_double', dqpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('dqp2_single', dqpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('dqp2_double', dqpti_double, suite: ['fortran', 'double'], is_parallel : false)

eqpt_single = executable('eqpt_single', 'src/eqp/eqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
eqpt_double = executable('eqpt_double', 'src/eqp/eqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
eqpti_single = executable('eqpti_single', 'src/eqp/eqpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
eqpti_double = executable('eqpti_double', 'src/eqp/eqpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('eqp1_single', eqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('eqp1_double', eqpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('eqp2_single', eqpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('eqp2_double', eqpti_double, suite: ['fortran', 'double'], is_parallel : false)

bgot_single = executable('bgot_single', 'src/bgo/bgot.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bgot_double = executable('bgot_double', 'src/bgo/bgot.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
bgoti_single = executable('bgoti_single', 'src/bgo/bgoti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bgoti_double = executable('bgoti_double', 'src/bgo/bgoti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('bgo1_single', bgot_single, suite: ['fortran', 'single'], is_parallel : false)
test('bgo1_double', bgot_double, suite: ['fortran', 'double'], is_parallel : false)
test('bgo2_single', bgoti_single, suite: ['fortran', 'single'], is_parallel : false)
test('bgo2_double', bgoti_double, suite: ['fortran', 'double'], is_parallel : false)

# Issue 64
# cllst_single = executable('cllst_single', 'src/clls/cllst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
# cllst_double = executable('cllst_double', 'src/clls/cllst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# cllsti_single = executable('cllsti_single', 'src/clls/cllsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
# cllsti_double = executable('cllsti_double', 'src/clls/cllsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# test('clls1_single', cllst_single, suite: ['fortran', 'single'], is_parallel : false)
# test('clls1_double', cllst_double, suite: ['fortran', 'double'], is_parallel : false)
# test('clls2_single', cllsti_single, suite: ['fortran', 'single'], is_parallel : false)
# test('clls2_double', cllsti_double, suite: ['fortran', 'double'], is_parallel : false)

nlst_single = executable('nlst_single', 'src/nls/nlst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
nlst_double = executable('nlst_double', 'src/nls/nlst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
nlsti_single = executable('nlsti_single', 'src/nls/nlsti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
nlsti_double = executable('nlsti_double', 'src/nls/nlsti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('nls1_single', nlst_single, suite: ['fortran', 'single'], is_parallel : false)
test('nls1_double', nlst_double, suite: ['fortran', 'double'], is_parallel : false)
test('nls2_single', nlsti_single, suite: ['fortran', 'single'], is_parallel : false)
test('nls2_double', nlsti_double, suite: ['fortran', 'double'], is_parallel : false)

bqpbt_single = executable('bqpbt_single', 'src/bqpb/bqpbt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bqpbt_double = executable('bqpbt_double', 'src/bqpb/bqpbt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
bqpbti_single = executable('bqpbti_single', 'src/bqpb/bqpbti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
bqpbti_double = executable('bqpbti_double', 'src/bqpb/bqpbti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('bqpb1_single', bqpbt_single, suite: ['fortran', 'single'], is_parallel : false)
test('bqpb1_double', bqpbt_double, suite: ['fortran', 'double'], is_parallel : false)
test('bqpb2_single', bqpbti_single, suite: ['fortran', 'single'], is_parallel : false)
test('bqpb2_double', bqpbti_double, suite: ['fortran', 'double'], is_parallel : false)

ccqpt_single = executable('ccqpt_single', 'src/ccqp/ccqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ccqpt_double = executable('ccqpt_double', 'src/ccqp/ccqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
ccqpti_single = executable('ccqpti_single', 'src/ccqp/ccqpti.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
ccqpti_double = executable('ccqpti_double', 'src/ccqp/ccqpti.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('ccqp1_single', ccqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('ccqp1_double', ccqpt_double, suite: ['fortran', 'double'], is_parallel : false)
test('ccqp2_single', ccqpti_single, suite: ['fortran', 'single'], is_parallel : false)
test('ccqp2_double', ccqpti_double, suite: ['fortran', 'double'], is_parallel : false)

# See issue #40 on GitHub
# pastixt_single = executable('pastixt_single', 'src/external/pastix/pastixt.F90', fortran_args : extra_args_single, link_with : [libspmf, libpastix_single], include_directories: libgalahad_include)
# pastixt_double = executable('pastixt_double', 'src/external/pastix/pastixt.F90', fortran_args : extra_args_double, link_with : [libspmf, libpastix_double], include_directories: libgalahad_include)
# test('pastix_single', pastixt_single, suite: ['fortran', 'single'], is_parallel : false)
# test('pastix_double', pastixt_double, suite: ['fortran', 'double'], is_parallel : false)

filtranet_single = executable('filtranet_single', 'src/filtrane/filtranet.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
filtranet_double = executable('filtranet_double', 'src/filtrane/filtranet.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('filtrane_single', filtranet_single, suite: ['fortran', 'single'], is_parallel : false)
test('filtrane_double', filtranet_double, suite: ['fortran', 'double'], is_parallel : false)

cdqpt_single = executable('cdqpt_single', 'src/cdqp/cdqpt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
cdqpt_double = executable('cdqpt_double', 'src/cdqp/cdqpt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('cdqp_single', cdqpt_single, suite: ['fortran', 'single'], is_parallel : false)
test('cdqp_double', cdqpt_double, suite: ['fortran', 'double'], is_parallel : false)

if build_ciface
  ugot_c_single = executable('ugot_c_single', 'src/ugo/C/ugot.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  ugot_c_double = executable('ugot_c_double', 'src/ugo/C/ugot.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('ugo_c_single', ugot_c_single, suite: ['C', 'single'], is_parallel : false)
  test('ugo_c_double', ugot_c_double, suite: ['C', 'double'], is_parallel : false)

  slst_c_single = executable('slst_c_single', 'src/sls/C/slst.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  slst_c_double = executable('slst_c_double', 'src/sls/C/slst.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('sls_c_single', slst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('sls_c_double', slst_c_double, suite: ['C', 'double'], is_parallel : false)

  slstf_c_single = executable('slstf_c_single', 'src/sls/C/slstf.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  slstf_c_double = executable('slstf_c_double', 'src/sls/C/slstf.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('sls2_c_single', slst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('sls2_c_double', slst_c_double, suite: ['C', 'double'], is_parallel : false)

  ulst_c_single = executable('ulst_c_single', 'src/uls/C/ulst.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  ulst_c_double = executable('ulst_c_double', 'src/uls/C/ulst.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('uls_c_single', ulst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('uls_c_double', ulst_c_double, suite: ['C', 'double'], is_parallel : false)

  ulstf_c_single = executable('ulstf_c_single', 'src/uls/C/ulstf.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  ulstf_c_double = executable('ulstf_c_double', 'src/uls/C/ulstf.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('uls2_c_single', ulst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('uls2_c_double', ulst_c_double, suite: ['C', 'double'], is_parallel : false)

  sblst_c_single = executable('sblst_c_single', 'src/sbls/C/sblst.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  sblst_c_double = executable('sblst_c_double', 'src/sbls/C/sblst.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('sbls_c_single', sblst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('sbls_c_double', sblst_c_double, suite: ['C', 'double'], is_parallel : false)

  sblstf_c_single = executable('sblstf_c_single', 'src/sbls/C/sblstf.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  sblstf_c_double = executable('sblstf_c_double', 'src/sbls/C/sblstf.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('sbls2_c_single', sblst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('sbls2_c_double', sblst_c_double, suite: ['C', 'double'], is_parallel : false)

  pslst_c_single = executable('pslst_c_single', 'src/psls/C/pslst.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  pslst_c_double = executable('pslst_c_double', 'src/psls/C/pslst.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('psls_c_single', pslst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('psls_c_double', pslst_c_double, suite: ['C', 'double'], is_parallel : false)

  pslstf_c_single = executable('pslstf_c_single', 'src/psls/C/pslstf.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  pslstf_c_double = executable('pslstf_c_double', 'src/psls/C/pslstf.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('psls2_c_single', pslst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('psls2_c_double', pslst_c_double, suite: ['C', 'double'], is_parallel : false)

  trst_c_single = executable('trst_c_single', 'src/trs/C/trst.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  trst_c_double = executable('trst_c_double', 'src/trs/C/trst.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('trs_c_single', trst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('trs_c_double', trst_c_double, suite: ['C', 'double'], is_parallel : false)

  trstf_c_single = executable('trstf_c_single', 'src/trs/C/trstf.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  trstf_c_double = executable('trstf_c_double', 'src/trs/C/trstf.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('trs2_c_single', trst_c_single, suite: ['C', 'single'], is_parallel : false)
  test('trs2_c_double', trst_c_double, suite: ['C', 'double'], is_parallel : false)
endif

#if build_ciface
  # TODO: test C interface
  # gltr, glrt, lstr, lsrt, l2rt, presolve, rpd, ugo
#endif

# Example programs

build_examples = get_option('examples')

if build_examples
  share_examples = 'share/examples'
  executable('amds', 'src/amd/amds.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bscs', 'src/bsc/bscs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('checks', 'src/check/checks.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('converts', 'src/convert/converts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('fdhs', 'src/fdh/fdhs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('fdhs2', 'src/fdh/fdhs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)  # FIXME
  executable('fits', 'src/fit/fits.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('gltrs', 'src/gltr/gltrs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('gltrs2', 'src/gltr/gltrs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('gltrs3', 'src/gltr/gltrs3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('glrts', 'src/glrt/glrts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('glss', 'src/gls/glss.f90', link_with : [libgalahad_hsl_double, libgalahad_double], install : true, install_dir : share_examples)
  executable('hashs', 'src/hash/hashs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lqrs', 'src/lqr/lqrs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lqrs2', 'src/lqr/lqrs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lqts', 'src/lqt/lqts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lqts2', 'src/lqt/lqts2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lsps', 'src/lsp/lsps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lstrs', 'src/lstr/lstrs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lstrs2', 'src/lstr/lstrs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lsrts', 'src/lsrt/lsrts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lsrts2', 'src/lsrt/lsrts2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('l2rts', 'src/l2rt/l2rts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('l2rts2', 'src/l2rt/l2rts2.f90', link_with : libgalahad, install : true, install_dir : share_examples)  # FIXME
  executable('lhss', 'src/lhs/lhss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lmss', 'src/lms/lmss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('miqrs', 'src/miqr/miqrs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('miqrs2', 'src/miqr/miqrs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('mops', 'src/mop/mops.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('nlpts', 'src/nlpt/nlpts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('presolves', 'src/presolve/presolves.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('qpps', 'src/qpp/qpps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('qpts', 'src/qpt/qpts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('rands', 'src/rand/rands.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('rpds', 'src/rpd/rpds.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('rootss', 'src/roots/rootss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('scales', 'src/scale/scales.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('scus', 'src/scu/scus.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('secs', 'src/sec/secs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('shas', 'src/sha/shas.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('silss', 'src/sils/silss.f90', link_with : [libgalahad_hsl_double, libgalahad_double], install : true, install_dir : share_examples)
  executable('smts', 'src/smt/smts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sorts', 'src/sort/sorts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ugos', 'src/ugo/ugos.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ugos2', 'src/ugo/ugos2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sls', 'src/sls/slss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('uls', 'src/uls/ulss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sbls', 'src/sbls/sblss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ir', 'src/ir/irs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('psls', 'src/psls/pslss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trs', 'src/trs/trss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trs2', 'src/trs/trss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('icfs', 'src/icfs/icfss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss', 'src/slls/sllss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss2', 'src/slls/sllss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss3', 'src/slls/sllss3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss4', 'src/slls/sllss4.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss5', 'src/slls/sllss5.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss6', 'src/slls/sllss6.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('sllss7', 'src/slls/sllss7.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cros', 'src/cro/cros.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cros2', 'src/cro/cros2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cros3', 'src/cro/cros3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('fdcs', 'src/fdc/fdcs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('wcps', 'src/wcp/wcps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('rqss', 'src/rqs/rqss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('rqss2', 'src/rqs/rqss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bllss', 'src/blls/bllss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bllss2', 'src/blls/bllss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bllss3', 'src/blls/bllss3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bqps', 'src/bqp/bqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bqps2', 'src/bqp/bqps2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bqps3', 'src/bqp/bqps3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cqps', 'src/cqp/cqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cqps2', 'src/cqp/cqps2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cqps3', 'src/cqp/cqps3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dlps', 'src/dlp/dlps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dqps', 'src/dqp/dqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dqps2', 'src/dqp/dqps2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dqps3', 'src/dqp/dqps3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('eqps', 'src/eqp/eqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('eqps2', 'src/eqp/eqps.options.F90', link_with : libgalahad_double, install : true, install_dir : share_examples, include_directories: libgalahad_include)
  executable('l1qp', 'src/l1qp/l1qps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dps', 'src/dps/dpss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dps2', 'src/dps/dpss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lancelot', 'src/lancelot/lancelots.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('lancelot_steering', 'src/lancelot/lancelot_steerings.F90', link_with : libgalahad_double, install : true, install_dir : share_examples, include_directories: libgalahad_include)
  executable('lancelot_simple', 'src/lancelot_simple/lancelot_simples.f90', link_with : libgalahad_double, install : true, install_dir : share_examples, include_directories: libgalahad_include)
  executable('filtrane', 'src/filtrane/filtranes.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trbs', 'src/trb/trbs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trbs2', 'src/trb/trbs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trbs3', 'src/trb/trbs3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('nlss', 'src/nls/nlss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('nlss2', 'src/nls/nlss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('nlss3', 'src/nls/nlss3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('ssids', 'src/ssids/ssidss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trus', 'src/tru/trus.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trus2', 'src/tru/trus2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trus3', 'src/tru/trus3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('trus4', 'src/tru/trus4.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bgos', 'src/bgo/bgos.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bgos2', 'src/bgo/bgos2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('arcs', 'src/arc/arcs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('arcs2', 'src/arc/arcs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('arcs3', 'src/arc/arcs3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # Issue 62
  # executable('arcs4', 'src/arc/arcs4.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dgo', 'src/dgo/dgos.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('dgo2', 'src/dgo/dgos2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # Issue 64
  # executable('cllss', 'src/clls/cllss.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('cllss2', 'src/clls/cllss2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # executable('cllss3', 'src/clls/cllss3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bqpbs', 'src/bqpb/bqpbs.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('bqpbs2', 'src/bqpb/bqpbs2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  # Issue 65
  # executable('bqpbs3', 'src/bqpb/bqpbs3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ccqps', 'src/ccqp/ccqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ccqps2', 'src/ccqp/ccqps2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('ccqps3', 'src/ccqp/ccqps3.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
  executable('cdqps', 'src/cdqp/cdqps.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
endif

if build_ciface
  ugos_c_single = executable('ugos_c_single', 'src/ugo/C/ugos.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  ugos_c_double = executable('ugos_c_double', 'src/ugo/C/ugos.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  test('ugos_c_single', ugos_c_single, suite: ['C', 'single'], is_parallel : false)
  test('ugos_c_double', ugos_c_double, suite: ['C', 'double'], is_parallel : false)
endif
