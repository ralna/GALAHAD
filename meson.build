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

libgalahad_hsl_src = []

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

libgalahad_hsl_src += [
  ad02_path / 'hsl_ad02s.f90',
  ad02_path / 'hsl_ad02d.f90',
]
libgalahad_hsl_src += [
  ma27_path / 'ma27s.f',
  ma27_path / 'ma27d.f',
]
libgalahad_hsl_src += [
  ma33_path / 'ma33s.f',
  ma33_path / 'ma33d.f',
  ma33_path / 'mc13s.f',
  ma33_path / 'mc13d.f',
  ma33_path / 'mc20s.f',
  ma33_path / 'mc20d.f',
  ma33_path / 'mc21s.f',
  ma33_path / 'mc21d.f',
]

libgalahad_hsl = library('galahad_hsl',
                         sources : libgalahad_hsl_src,
                         link_with : libgalahad_link_with,
                         dependencies : blas_dep + lapack_dep,
                         install : true)
libgalahad_link_with += [libgalahad_hsl]

## 1. the following modules have no dependencies inside libgalahad; they appear in alphabetical order
subdir('src/clock')
subdir('src/checkpoint')
subdir('src/common')
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
# subdir('src/ptrans') # needs CUTEst interface
subdir('src/qpp')
subdir('src/scale')
subdir('src/scaling')

# packages that require hslarchive_galahad to be nontrivial
subdir('src/gls')
subdir('src/sils')

# ir  => needs SLS
# TODO: ssids
# sls => needs HSL

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

# Executables

# Require GALAHAD_KINDS
# buildspec = executable('buildspec', 'src/buildspec/buildspec.F90', include_directories: libgalahad_include, install : true)

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

glrtt_single = executable('glrtt_single', 'src/glrt/glrtt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
glrtt_double = executable('glrtt_double', 'src/glrt/glrtt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('glrt_single', glrtt_single, suite: ['fortran', 'single'], is_parallel : false)
test('glrt_double', glrtt_double, suite: ['fortran', 'double'], is_parallel : false)

glst_single = executable('glst_single', 'src/gls/glst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
glst_double = executable('glst_double', 'src/gls/glst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('gls_single', glst_single, suite: ['fortran', 'single'], is_parallel : false)
test('gls_double', glst_double, suite: ['fortran', 'double'], is_parallel : false)

hasht_single = executable('hasht_single', 'src/hash/hasht.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
hasht_double = executable('hasht_double', 'src/hash/hasht.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('hash_single', hasht_single, suite: ['fortran', 'single'], is_parallel : false)
test('hash_double', hasht_double, suite: ['fortran', 'double'], is_parallel : false)

lqrt_single = executable('lqrt_single', 'src/lqr/lqrt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
lqrt_double = executable('lqrt_double', 'src/lqr/lqrt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('lqr_single', lqrt_single, suite: ['fortran', 'single'], is_parallel : false)
test('lqr_double', lqrt_double, suite: ['fortran', 'double'], is_parallel : false)

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

# Issue with BLAS/LAPACK symbols "isamax_" and "isdmax_"
# mopt_single = executable('mopt_single', 'src/mop/mopt.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
# mopt_double = executable('mopt_double', 'src/mop/mopt.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# test('mop_single', mopt_single, suite: ['fortran', 'single'], is_parallel : false)
# test('mop_double', mopt_double, suite: ['fortran', 'double'], is_parallel : false)

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

# FIXME: install data files
# rpdt = executable('rpdt', 'src/rpd/rpdt.F90', link_with : libgalahad_double)
# rpdti = executable('rpdti', 'src/rpd/rpdti.F90', link_with : libgalahad_double)
# test('rpd', rpdt)
# test('rpdi', rpdti)

rootst_single = executable('rootst_single', 'src/roots/rootst.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
rootst_double = executable('rootst_double', 'src/roots/rootst.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('roots_single', rootst_single, suite: ['fortran', 'single'], is_parallel : false)
test('roots_double', rootst_double, suite: ['fortran', 'double'], is_parallel : false)

# scut_single = executable('scut_single', 'src/scu/scut.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
# scut_double = executable('scut_double', 'src/scu/scut.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
# test('scu_single', scut_single, suite: ['fortran', 'single'], is_parallel : false)
# test('scu_double', scut_double, suite: ['fortran', 'double'], is_parallel : false)

scalet_single = executable('scalet_single', 'src/scale/scalet.F90', fortran_args : extra_args_single, link_with : libgalahad_single, include_directories: libgalahad_include)
scalet_double = executable('scalet_double', 'src/scale/scalet.F90', fortran_args : extra_args_double, link_with : libgalahad_double, include_directories: libgalahad_include)
test('scale_single', scalet_single, suite: ['fortran', 'single'], is_parallel : false)
test('scale_double', scalet_double, suite: ['fortran', 'double'], is_parallel : false)

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
test('ugo_single', ugot_single, suite: ['fortran', 'single'], is_parallel : false)
test('ugo_double', ugot_double, suite: ['fortran', 'double'], is_parallel : false)

if build_ciface
  # ugot_c_single = executable('ugot_c_single', 'src/ugo/C/ugot.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  # ugot_c_double = executable('ugot_c_double', 'src/ugo/C/ugot.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  # test('ugot_c_single', ugot_c_single, suite: ['C', 'single'], is_parallel : false)
  # test('ugot_c_double', ugot_c_double, suite: ['C', 'double'], is_parallel : false)
endif

#if build_ciface
  # TODO: test C interface
  # gltr, glrt, lstr, lsrt, l2rt, presolve, rpd, ugo
#endif

# Example programs

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
executable('glss', 'src/gls/glss.f90', link_with : [libgalahad_hsl, libgalahad_double], install : true, install_dir : share_examples)
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
executable('silss', 'src/sils/silss.f90', link_with : [libgalahad_hsl, libgalahad_double], install : true, install_dir : share_examples)
executable('smts', 'src/smt/smts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
executable('sorts', 'src/sort/sorts.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
executable('ugos', 'src/ugo/ugos.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)
executable('ugos2', 'src/ugo/ugos2.f90', link_with : libgalahad_double, install : true, install_dir : share_examples)

if build_ciface
  # ugos_c_single = executable('ugos_c_single', 'src/ugo/C/ugos.c', c_args : extra_args_single, link_with : libgalahad_c_single, include_directories: libgalahad_include)
  # ugos_c_double = executable('ugos_c_double', 'src/ugo/C/ugos.c', c_args : extra_args_double, link_with : libgalahad_c_double, include_directories: libgalahad_include)
  # test('ugos_c_single', ugos_c_single, suite: ['C', 'single'], is_parallel : false)
  # test('ugos_c_double', ugos_c_double, suite: ['C', 'double'], is_parallel : false)
endif
