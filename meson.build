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

bingalahad_src = []

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

libgalahad_include = ['include', 'src/dum/include', 'src/ampl']

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

# Fortran modules
if install_modules
  meson.add_install_script(script_modules)
endif

# Executables
foreach val: bingalahad_src
  binname = val[0]
  binfile = val[1]

  if build_single
    executable(binname+'_single', binfile,
               fortran_args : extra_args_single,
               link_with : libgalahad_single,
               link_language : 'fortran',
               include_directories: libgalahad_include,
               install : true)
  endif
  if build_double
    executable(binname+'_double', binfile,
               fortran_args : extra_args_double,
               link_with : libgalahad_double,
               link_language : 'fortran',
               include_directories: libgalahad_include,
               install : true)
  endif
endforeach

# Unit tests

if build_tests

  fortran_tests_folder = 'share/tests/fortran'
  fortran_tests = []
  if build_single
    fortran_tests += [['single', extra_args_single, libgalahad_single]]
  endif
  if build_double
    fortran_tests += [['double', extra_args_double, libgalahad_double]]
  endif

  foreach val: fortran_tests
    amdt = executable('amdt_'+val[0], 'src/amd/amdt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('amdt_'+val[0], amdt, suite: ['amd', 'fortran', val[0]], is_parallel : false)

    bsct = executable('bsct_'+val[0], 'src/bsc/bsct.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('bsct_'+val[0], bsct, suite: ['bsc', 'fortran', val[0]], is_parallel : false)

    checkt = executable('checkt_'+val[0], 'src/check/checkt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('checkt_'+val[0], checkt, suite: ['check', 'fortran', val[0]], is_parallel : false)

    convertt = executable('convertt_'+val[0], 'src/convert/convertt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('convertt_'+val[0], convertt, suite: ['convert', 'fortran', val[0]], is_parallel : false)

    fdht = executable('fdht_'+val[0], 'src/fdh/fdht.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('fdht_'+val[0], fdht, suite: ['fdh', 'fortran', val[0]], is_parallel : false)

    fitt = executable('fitt_'+val[0], 'src/fit/fitt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('fitt_'+val[0], fitt, suite: ['fit', 'fortran', val[0]], is_parallel : false)

    gltrt = executable('gltrt_'+val[0], 'src/gltr/gltrt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    gltrti = executable('gltrti_'+val[0], 'src/gltr/gltrti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('gltrt_'+val[0], gltrt, suite: ['gltr', 'fortran', val[0]], is_parallel : false)
    test('gltrti_'+val[0], gltrti, suite: ['gltr', 'fortran', val[0]], is_parallel : false)

    dgot = executable('dgot_'+val[0], 'src/dgo/dgot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    dgoti = executable('dgoti_'+val[0], 'src/dgo/dgoti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('dgot_'+val[0], dgot, suite: ['dgo', 'fortran', val[0]], is_parallel : false)
    test('dgoti_'+val[0], dgoti, suite: ['dgo', 'fortran', val[0]], is_parallel : false)

    glrtt = executable('glrtt_'+val[0], 'src/glrt/glrtt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('glrtt_'+val[0], glrtt, suite: ['glrt', 'fortran', val[0]], is_parallel : false)

    glst = executable('glst_'+val[0], 'src/gls/glst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('glst_'+val[0], glst, suite: ['gls', 'fortran', val[0]], is_parallel : false)

    crot = executable('crot_'+val[0], 'src/cro/crot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('crot_'+val[0], crot, suite: ['cro', 'fortran', val[0]], is_parallel : false)

    hasht = executable('hasht_'+val[0], 'src/hash/hasht.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('hasht_'+val[0], hasht, suite: ['hash', 'fortran', val[0]], is_parallel : false)

    lqrt = executable('lqrt_'+val[0], 'src/lqr/lqrt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lqrt_'+val[0], lqrt, suite: ['lqr', 'fortran', val[0]], is_parallel : false)

    lancelot_simplet = executable('lancelot_simplet_'+val[0], 'src/lancelot_simple/lancelot_simplet.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lancelot_simplet_'+val[0], lancelot_simplet, suite: ['lancelot_simple', 'fortran', val[0]], is_parallel : false)

    lqtt = executable('lqtt_'+val[0], 'src/lqt/lqtt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lqtt2 = executable('lqtt2_'+val[0], 'src/lqt/lqtt2.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lqtt_'+val[0], lqtt, suite: ['lqt', 'fortran', val[0]], is_parallel : false)
    test('lqtt2_'+val[0], lqtt2, suite: ['lqt', 'fortran', val[0]], is_parallel : false)

    lspt = executable('lspt_'+val[0], 'src/lsp/lspt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lspt_'+val[0], lspt, suite: ['lsp', 'fortran', val[0]], is_parallel : false)

    lstrt = executable('lstrt_'+val[0], 'src/lstr/lstrt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lstrti = executable('lstrti_'+val[0], 'src/lstr/lstrti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lstrt_'+val[0], lstrt, suite: ['lstr', 'fortran', val[0]], is_parallel : false)
    test('lstrti_'+val[0], lstrti, suite: ['lstr', 'fortran', val[0]], is_parallel : false)

    fdct = executable('fdct_'+val[0], 'src/fdc/fdct.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    fdcti = executable('fdcti_'+val[0], 'src/fdc/fdcti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('fdct_'+val[0], fdct, suite: ['fdc', 'fortran', val[0]], is_parallel : false)
    test('fdcti_'+val[0], fdcti, suite: ['fdc', 'fortran', val[0]], is_parallel : false)

    lsrtt = executable('lsrtt_'+val[0], 'src/lsrt/lsrtt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lsrtti = executable('lsrtti_'+val[0], 'src/lsrt/lsrtti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lsrtt_'+val[0], lsrtt, suite: ['lsrt', 'fortran', val[0]], is_parallel : false)
    test('lsrtti_'+val[0], lsrtti, suite: ['lsrt', 'fortran', val[0]], is_parallel : false)

    l2rtt  = executable('l2rtt_'+val[0] , 'src/l2rt/l2rtt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    l2rtti = executable('l2rtti_'+val[0], 'src/l2rt/l2rtti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('l2rtt_'+val[0], l2rtt, suite: ['l2rt', 'fortran', val[0]], is_parallel : false)
    test('l2rtti_'+val[0], l2rtti, suite: ['l2rt', 'fortran', val[0]], is_parallel : false)

    rqst  = executable('rqst_'+val[0] , 'src/rqs/rqst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    rqsti = executable('rqsti_'+val[0], 'src/rqs/rqsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('rqst_'+val[0], rqst, suite: ['rqs', 'fortran', val[0]], is_parallel : false)
    test('rqsti_'+val[0], rqsti, suite: ['rqs', 'fortran', val[0]], is_parallel : false)

    lhst = executable('lhst_'+val[0], 'src/lhs/lhst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lhst_'+val[0], lhst, suite: ['lhs', 'fortran', val[0]], is_parallel : false)

    lmst = executable('lmst_'+val[0], 'src/lms/lmst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lmst_'+val[0], lmst, suite: ['lms', 'fortran', val[0]], is_parallel : false)

    miqrt = executable('miqrt_'+val[0], 'src/miqr/miqrt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('miqrt_'+val[0], miqrt, suite: ['miqr', 'fortran', val[0]], is_parallel : false)

    mopt = executable('mopt_'+val[0], 'src/mop/mopt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('mopt_'+val[0], mopt, suite: ['mop', 'fortran', val[0]], is_parallel : false)

    dpst = executable('dpst_'+val[0], 'src/dps/dpst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('dpst_'+val[0], dpst, suite: ['dps', 'fortran', val[0]], is_parallel : false)

    sllst = executable('sllst_'+val[0], 'src/slls/sllst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    sllsti = executable('sllsti_'+val[0], 'src/slls/sllsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('sllst_'+val[0], sllst, suite: ['slls', 'fortran', val[0]], is_parallel : false)
    test('sllsti_'+val[0], sllsti, suite: ['slls', 'fortran', val[0]], is_parallel : false)

    nlptt = executable('nlptt_'+val[0], 'src/nlpt/nlptt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('nlptt_'+val[0], nlptt, suite: ['nlpt', 'fortran', val[0]], is_parallel : false)

    presolvet = executable('presolvet_'+val[0], 'src/presolve/presolvet.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    presolveti = executable('presolveti_'+val[0], 'src/presolve/presolveti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('presolvet_'+val[0], presolvet, suite: ['presolve', 'fortran', val[0]], is_parallel : false)
    test('presolveti_'+val[0], presolveti, suite: ['presolve', 'fortran', val[0]], is_parallel : false)

    qppt = executable('qppt_'+val[0], 'src/qpp/qppt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qppt_'+val[0], qppt, suite: ['qpp', 'fortran', val[0]], is_parallel : false)

    qptt = executable('qptt_'+val[0], 'src/qpt/qptt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qptt_'+val[0], qptt, suite: ['qpt', 'fortran', val[0]], is_parallel : false)

    randt = executable('randt_'+val[0], 'src/rand/randt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('randt_'+val[0], randt, suite: ['rand', 'fortran', val[0]], is_parallel : false)

    rpdt = executable('rpdt_'+val[0], 'src/rpd/rpdt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    rpdti = executable('rpdti_'+val[0], 'src/rpd/rpdti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('rpdt_'+val[0], rpdt, suite: ['rpd', 'fortran', val[0]], is_parallel : false)
    test('rpdti_'+val[0], rpdti, suite: ['rpd', 'fortran', val[0]], is_parallel : false)

    bllst = executable('bllst_'+val[0], 'src/blls/bllst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    bllsti = executable('bllsti_'+val[0], 'src/blls/bllsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('bllst_'+val[0], bllst, suite: ['blls', 'fortran', val[0]], is_parallel : false)
    test('bllsti_'+val[0], bllsti, suite: ['blls', 'fortran', val[0]], is_parallel : false)

    trut = executable('trut_'+val[0], 'src/tru/trut.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    truti = executable('truti_'+val[0], 'src/tru/truti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('trut_'+val[0], trut, suite: ['tru', 'fortran', val[0]], is_parallel : false)
    test('truti_'+val[0], truti, suite: ['tru', 'fortran', val[0]], is_parallel : false)

    rootst = executable('rootst_'+val[0], 'src/roots/rootst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('rootst_'+val[0], rootst, suite: ['roots', 'fortran', val[0]], is_parallel : false)

    scut = executable('scut_'+val[0], 'src/scu/scut.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('scut_'+val[0], scut, suite: ['scu', 'fortran', val[0]], is_parallel : false)

    scalet = executable('scalet_'+val[0], 'src/scale/scalet.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('scalet_'+val[0], scalet, suite: ['scale', 'fortran', val[0]], is_parallel : false)

    sect = executable('sect_'+val[0], 'src/sec/sect.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('sect_'+val[0], sect, suite: ['sec', 'fortran', val[0]], is_parallel : false)

    shat = executable('shat_'+val[0], 'src/sha/shat.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('shat_'+val[0], shat, suite: ['sha', 'fortran', val[0]], is_parallel : false)

    silst = executable('silst_'+val[0], 'src/sils/silst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('silst_'+val[0], silst, suite: ['sils', 'fortran', val[0]], is_parallel : false)

    smtt = executable('smtt_'+val[0], 'src/smt/smtt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('smtt_'+val[0], smtt, suite: ['smt', 'fortran', val[0]], is_parallel : false)

    sortt = executable('sortt_'+val[0], 'src/sort/sortt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('sortt_'+val[0], sortt, suite: ['sort', 'fortran', val[0]], is_parallel : false)

    ugot = executable('ugot_'+val[0], 'src/ugo/ugot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('ugot_'+val[0], ugot, suite: ['ugo', 'fortran', val[0]], is_parallel : false)

    mkl_pardisot = executable('mkl_pardisot_'+val[0], 'src/external/mkl_pardiso/mkl_pardisot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('mkl_pardisot_'+val[0], mkl_pardisot, suite: ['mkl_pardiso', 'fortran', val[0]], is_parallel : false)

    mumpst = executable('mumpst_'+val[0], 'src/external/mumps/mumpst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('mumpst_'+val[0], mumpst, suite: ['mumps', 'fortran', val[0]], is_parallel : false)

    pardisot = executable('pardisot_'+val[0], 'src/external/pardiso/pardisot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('pardisot_'+val[0], pardisot, suite: ['pardiso', 'fortran', val[0]], is_parallel : false)

    wsmpt = executable('wsmpt_'+val[0], 'src/external/wsmp/wsmpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('wsmpt_'+val[0], wsmpt, suite: ['wsmp', 'fortran', val[0]], is_parallel : false)

    pastixt = executable('pastixt_'+val[0], 'src/external/pastix/pastixt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('pastixt_'+val[0], pastixt, suite: ['pastix', 'fortran', val[0]], is_parallel : false)

    slst = executable('slst_'+val[0], ['src/sls/slst.F90', 'src/dum/mpi.F90'], fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    slsti = executable('slsti_'+val[0], 'src/sls/slsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('slst_'+val[0], slst, suite: ['sls', 'fortran', val[0]], is_parallel : false)
    test('slsti_'+val[0], slsti, suite: ['sls', 'fortran', val[0]], is_parallel : false)

    ulst = executable('ulst_'+val[0], 'src/uls/ulst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    ulsti = executable('ulsti_'+val[0], 'src/uls/ulsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('ulst_'+val[0], ulst, suite: ['uls', 'fortran', val[0]], is_parallel : false)
    test('ulsti_'+val[0], ulsti, suite: ['uls', 'fortran', val[0]], is_parallel : false)

    irt = executable('irt_'+val[0], 'src/ir/irt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('ir_'+val[0], irt, suite: ['ir', 'fortran', val[0]], is_parallel : false)

    icfst = executable('icfst_'+val[0], 'src/icfs/icfst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('icfst_'+val[0], icfst, suite: ['icfs', 'fortran', val[0]], is_parallel : false)

    trst = executable('trst_'+val[0], 'src/trs/trst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    trsti = executable('trsti_'+val[0], 'src/trs/trsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('trst_'+val[0], trst, suite: ['trs', 'fortran', val[0]], is_parallel : false)
    test('trsti_'+val[0], trsti, suite: ['trs', 'fortran', val[0]], is_parallel : false)

    trbt = executable('trbt_'+val[0], 'src/trb/trbt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    trbti = executable('trbti_'+val[0], 'src/trb/trbti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('trbt_'+val[0], trbt, suite: ['trb', 'fortran', val[0]], is_parallel : false)
    test('trbti_'+val[0], trbti, suite: ['trb', 'fortran', val[0]], is_parallel : false)

    wcpt = executable('wcpt_'+val[0], 'src/wcp/wcpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    wcpti = executable('wcpti_'+val[0], 'src/wcp/wcpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('wcpt_'+val[0], wcpt, suite: ['wcp', 'fortran', val[0]], is_parallel : false)
    test('wcpti_'+val[0], wcpti, suite: ['wcp', 'fortran', val[0]], is_parallel : false)

    pslst = executable('pslst_'+val[0], 'src/psls/pslst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    pslsti = executable('pslsti_'+val[0], 'src/psls/pslsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('pslst_'+val[0], pslst, suite: ['psls', 'fortran', val[0]], is_parallel : false)
    test('pslsti_'+val[0], pslsti, suite: ['psls', 'fortran', val[0]], is_parallel : false)

    lsqpt = executable('lsqpt_'+val[0], 'src/lsqp/lsqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lsqpti = executable('lsqpti_'+val[0], 'src/lsqp/lsqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lsqpt_'+val[0], lsqpt, suite: ['lsqp', 'fortran', val[0]], is_parallel : false)
    test('lsqpti_'+val[0], lsqpti, suite: ['lsqp', 'fortran', val[0]], is_parallel : false)

    lpat = executable('lpat_'+val[0], 'src/lpa/lpat.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lpati = executable('lpati_'+val[0], 'src/lpa/lpati.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lpat_'+val[0], lpat, suite: ['lpa', 'fortran', val[0]], is_parallel : false)
    test('lpati_'+val[0], lpati, suite: ['lpa', 'fortran', val[0]], is_parallel : false)

    lpbt = executable('lpbt_'+val[0], 'src/lpb/lpbt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lpbti = executable('lpbti_'+val[0], 'src/lpb/lpbti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lpbt_'+val[0], lpbt, suite: ['lpb', 'fortran', val[0]], is_parallel : false)
    test('lpbti_'+val[0], lpbti, suite: ['lpb', 'fortran', val[0]], is_parallel : false)

    arct = executable('arct_'+val[0], 'src/arc/arct.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    arcti = executable('arcti_'+val[0], 'src/arc/arcti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('arct_'+val[0], arct, suite: ['arc', 'fortran', val[0]], is_parallel : false)
    test('arcti_'+val[0], arcti, suite: ['arc', 'fortran', val[0]], is_parallel : false)

    sblst = executable('sblst_'+val[0], 'src/sbls/sblst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    sblsti = executable('sblsti_'+val[0], 'src/sbls/sblsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('sblst_'+val[0], sblst, suite: ['sbls', 'fortran', val[0]], is_parallel : false)
    test('sblsti_'+val[0], sblsti, suite: ['sbls', 'fortran', val[0]], is_parallel : false)

    cqpt = executable('cqpt_'+val[0], 'src/cqp/cqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    cqpti = executable('cqpti_'+val[0], 'src/cqp/cqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('cqpt_'+val[0], cqpt, suite: ['cqp', 'fortran', val[0]], is_parallel : false)
    test('cqpti_'+val[0], cqpti, suite: ['cqp', 'fortran', val[0]], is_parallel : false)

    bqpt = executable('bqpt_'+val[0], 'src/bqp/bqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    bqpti = executable('bqpti_'+val[0], 'src/bqp/bqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('bqpt_'+val[0], bqpt, suite: ['bqp', 'fortran', val[0]], is_parallel : false)
    test('bqpti_'+val[0], bqpti, suite: ['bqp', 'fortran', val[0]], is_parallel : false)

    lancelott = executable('lancelott_'+val[0], 'src/lancelot/lancelott.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    lancelot_steeringt = executable('lancelot_steeringt_'+val[0], 'src/lancelot/lancelot_steeringt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('lancelott_'+val[0], lancelott, suite: ['lancelot', 'fortran', val[0]], is_parallel : false)
    test('lancelot_steeringt_'+val[0], lancelot_steeringt, suite: ['lancelot', 'fortran', val[0]], is_parallel : false)

    llstt = executable('llstt_'+val[0], 'src/llst/llstt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('llstt_'+val[0], llstt, suite: ['llst', 'fortran', val[0]], is_parallel : false)

    l1qpt = executable('l1qpt_'+val[0], 'src/l1qp/l1qpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('l1qpt_'+val[0], l1qpt, suite: ['llqp', 'fortran', val[0]], is_parallel : false)

    dlpt = executable('dlpt_'+val[0], 'src/dlp/dlpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('dlpt_'+val[0], dlpt, suite: ['qlp', 'fortran', val[0]], is_parallel : false)

    dqpt = executable('dqpt_'+val[0], 'src/dqp/dqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    dqpti = executable('dqpti_'+val[0], 'src/dqp/dqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('dqpt_'+val[0], dqpt, suite: ['dqp', 'fortran', val[0]], is_parallel : false)
    test('dqpti_'+val[0], dqpti, suite: ['dqp', 'fortran', val[0]], is_parallel : false)

    eqpt = executable('eqpt_'+val[0], 'src/eqp/eqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    eqpti = executable('eqpti_'+val[0], 'src/eqp/eqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('eqpt_'+val[0], eqpt, suite: ['eqp', 'fortran', val[0]], is_parallel : false)
    test('eqpti_'+val[0], eqpti, suite: ['eqp', 'fortran', val[0]], is_parallel : false)

    bgot = executable('bgot_'+val[0], 'src/bgo/bgot.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    bgoti = executable('bgoti_'+val[0], 'src/bgo/bgoti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('bgot_'+val[0], bgot, suite: ['bgo', 'fortran', val[0]], is_parallel : false)
    test('bgoti_'+val[0], bgoti, suite: ['bgo', 'fortran', val[0]], is_parallel : false)

    nlst = executable('nlst_'+val[0], 'src/nls/nlst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    nlsti = executable('nlsti_'+val[0], 'src/nls/nlsti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('nlst_'+val[0], nlst, suite: ['nls', 'fortran', val[0]], is_parallel : false)
    test('nlsti_'+val[0], nlsti, suite: ['nls', 'fortran', val[0]], is_parallel : false)

    bqpbt = executable('bqpbt_'+val[0], 'src/bqpb/bqpbt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    bqpbti = executable('bqpbti_'+val[0], 'src/bqpb/bqpbti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('bqpbt_'+val[0], bqpbt, suite: ['bqpb', 'fortran', val[0]], is_parallel : false)
    test('bqpbti_'+val[0], bqpbti, suite: ['bqpb', 'fortran', val[0]], is_parallel : false)

    ccqpt = executable('ccqpt_'+val[0], 'src/ccqp/ccqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    ccqpti = executable('ccqpti_'+val[0], 'src/ccqp/ccqpti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('ccqpt_'+val[0], ccqpt, suite: ['ccqp', 'fortran', val[0]], is_parallel : false)
    test('ccqpti_'+val[0], ccqpti, suite: ['ccqp', 'fortran', val[0]], is_parallel : false)

    qpat = executable('qpat_'+val[0], 'src/qpa/qpat.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    qpati = executable('qpati_'+val[0], 'src/qpa/qpati.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qpat_'+val[0], qpat, suite: ['qpa', 'fortran', val[0]], is_parallel : false)
    test('qpati_'+val[0], qpati, suite: ['qpa', 'fortran', val[0]], is_parallel : false)

    qpbt = executable('qpbt_'+val[0], 'src/qpb/qpbt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    qpbti = executable('qpbti_'+val[0], 'src/qpb/qpbti.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qpbt_'+val[0], qpbt, suite: ['qpb', 'fortran', val[0]], is_parallel : false)
    test('qpbti_'+val[0], qpbti, suite: ['qpb', 'fortran', val[0]], is_parallel : false)

    filtranet = executable('filtranet_'+val[0], 'src/filtrane/filtranet.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('filtranet_'+val[0], filtranet, suite: ['filtrane', 'fortran', val[0]], is_parallel : false)

    cdqpt = executable('cdqpt_'+val[0], 'src/cdqp/cdqpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('cdqpt_'+val[0], cdqpt, suite: ['cdqp', 'fortran', val[0]], is_parallel : false)

    qpt = executable('qpt_'+val[0], 'src/qp/qpt.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qpt_'+val[0], qpt, suite: ['qp', 'fortran', val[0]], is_parallel : false)

    qpct = executable('qpct_'+val[0], 'src/qpc/qpct.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
    test('qpct_'+val[0], qpct, suite: ['qpc', 'fortran', val[0]], is_parallel : false)

    if build_ssids
      ssidst = executable('ssidst_'+val[0], 'src/ssids/ssidst.F90', fortran_args : val[1], link_with : val[2], dependencies : libgalahad_deps, link_language: 'fortran', include_directories: libgalahad_include)
      test('ssidst_'+val[0], ssidst, suite: ['ssids', 'fortran', val[0]], is_parallel : false)
    endif
  endforeach
endif

if build_tests and build_ciface

  c_tests_folder = 'share/tests/C'
  c_tests = []
  if build_single
    c_tests += [['single', extra_args_single, libgalahad_single]]
  endif
  if build_double
    c_tests += [['double', extra_args_double, libgalahad_double]]
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

  fortran_examples_folder = 'examples/Fortran'

  foreach example: galahad_examples
    executable(example[0], example[1], link_with : libgalahad_double, dependencies : libgalahad_deps, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
  endforeach
endif
# executable('ma48ss', 'src/external/ma48/ma48ss.f90', link_with : libgalahad_hsl_single, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)
# executable('ma48ds', 'src/external/ma48/ma48ds.f90', link_with : libgalahad_hsl_double, link_language: 'fortran', install : true, install_dir : fortran_examples_folder)

if build_examples and build_ciface

  c_examples_folder = 'examples/C'
  precisions = []
  if build_single
    precisions += [['_single', extra_args_single, libgalahad_single]]
  endif
  if build_double
    precisions += [['_double', extra_args_double, libgalahad_double]]
  endif

  foreach example: galahad_c_examples
    foreach val: precisions
      executable(example[0]+val[0], example[1], c_args : val[1], link_with : val[2], dependencies: lm, link_language: 'c', include_directories: libgalahad_include, install : true, install_dir : c_examples_folder)
    endforeach
  endforeach
endif
