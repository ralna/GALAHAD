project('GALAHAD', 'fortran')

# Each subdir() command includes a meson.build in a subdirectory
# that adds to the list of source files.

# libgalahad_blas, libgalahad_lapack

libgalahad_blas_src = []
libgalahad_lapack_src = []
subdir('src/lapack')
libgalahad_blas = library('galahad_blas', sources : libgalahad_blas_src, install : true)
libgalahad_lapack = library('galahad_lapack', sources : libgalahad_lapack_src, link_with : libgalahad_blas, install : true)

# libgalahad

libgalahad_src = []
libgalahad_c_src = []

## 1. the following modules have no dependencies inside libgalahad; they appear in alphabetical order
subdir('src/clock')
subdir('src/checkpoint')
subdir('src/common')
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

# gls => needs MA33
# TODO: hsl
# ir  => needs SLS
# sils => needs MA27
# TODO: ssids
# sls => needs HSL

libgalahad = library('galahad', sources : libgalahad_src, link_with : [libgalahad_blas, libgalahad_lapack], install : true)
build_ciface = get_option('ciface')
if build_ciface
  libgalahad_c = library('galahad_c', sources : libgalahad_c_src, link_with : libgalahad, install : true)
endif

# Executables

buildspec_src = 'src/buildspec/buildspec.f90'
buildspec = executable('buildspec', buildspec_src, link_with : libgalahad, install : true)

# Example programs

share_examples = 'share/examples'
executable('amds', 'src/amd/amds.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('bscs', 'src/bsc/bscs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('checks', 'src/check/checks.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('converts', 'src/convert/converts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('fdhs', 'src/fdh/fdhs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('fdhs2', 'src/fdh/fdhs2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('fits', 'src/fit/fits.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('gltrs', 'src/gltr/gltrs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('gltrs2', 'src/gltr/gltrs2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('gltrs3', 'src/gltr/gltrs3.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('glrts', 'src/glrt/glrts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('hashs', 'src/hash/hashs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lqrs', 'src/lqr/lqrs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lqrs2', 'src/lqr/lqrs2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lqts', 'src/lqt/lqts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lqts2', 'src/lqt/lqts2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lsps', 'src/lsp/lsps.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lstrs', 'src/lstr/lstrs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lstrs2', 'src/lstr/lstrs2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lsrts', 'src/lsrt/lsrts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lsrts2', 'src/lsrt/lsrts2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('l2rts', 'src/l2rt/l2rts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
# executable('l2rts2', 'src/l2rt/l2rts2.f90', link_with : libgalahad, install : true, install_dir : share_examples)  # FIXME
executable('lhss', 'src/lhs/lhss.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('lmss', 'src/lms/lmss.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('miqrs', 'src/miqr/miqrs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('miqrs2', 'src/miqr/miqrs2.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('mops', 'src/mop/mops.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('nlpts', 'src/nlpt/nlpts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('presolves', 'src/presolve/presolves.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('qpps', 'src/qpp/qpps.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('qpts', 'src/qpt/qpts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('rands', 'src/rand/rands.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('rpds', 'src/rpd/rpds.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('rootss', 'src/roots/rootss.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('scales', 'src/scale/scales.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('scus', 'src/scu/scus.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('secs', 'src/sec/secs.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('shas', 'src/sha/shas.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('smts', 'src/smt/smts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('sorts', 'src/sort/sorts.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('ugos', 'src/ugo/ugos.f90', link_with : libgalahad, install : true, install_dir : share_examples)
executable('ugos2', 'src/ugo/ugos2.f90', link_with : libgalahad, install : true, install_dir : share_examples)

# Unit tests

amdt = executable('amdt', 'src/amd/amdt.f90', link_with : libgalahad)
test('amd', amdt)

bsct = executable('bsct', 'src/bsc/bsct.f90', link_with : libgalahad)
test('bsc', bsct)

checkt = executable('checkt', 'src/check/checkt.f90', link_with : libgalahad)
test('check', checkt)

convertt = executable('convertt', 'src/convert/convertt.f90', link_with : libgalahad)
test('convert', convertt)

fdht = executable('fdht', 'src/fdh/fdht.f90', link_with : libgalahad)
test('fdh', fdht)

fitt = executable('fitt', 'src/fit/fitt.f90', link_with : libgalahad)
test('fit', fitt)

gltrt = executable('gltrt', 'src/gltr/gltrt.f90', link_with : libgalahad)
gltrti = executable('gltrti', 'src/gltr/gltrti.f90', link_with : libgalahad)
test('gltr', gltrt)
test('gltri', gltrti)

glrtt = executable('glrtt', 'src/glrt/glrtt.f90', link_with : libgalahad)
test('glrt', glrtt)

hasht = executable('hasht', 'src/hash/hasht.f90', link_with : libgalahad)
test('hash', hasht)

lqrt = executable('lqrt', 'src/lqr/lqrt.f90', link_with : libgalahad)
test('lqr', lqrt)

lqtt = executable('lqtt', 'src/lqt/lqtt.f90', link_with : libgalahad)
lqtt2 = executable('lqtt2', 'src/lqt/lqtt2.f90', link_with : libgalahad)
test('lqt', lqtt)
test('lqt2', lqtt2)

lspt = executable('lspt', 'src/lsp/lspt.f90', link_with : libgalahad)
test('lsp', lspt)

lstrt = executable('lstrt', 'src/lstr/lstrt.f90', link_with : libgalahad)
lstrti = executable('lstrti', 'src/lstr/lstrt.f90', link_with : libgalahad)
test('lstr', lstrt)
test('lstri', lstrti)

lsrtt = executable('lsrtt', 'src/lsrt/lsrtt.f90', link_with : libgalahad)
lsrtti = executable('lsrtti', 'src/lsrt/lsrtti.f90', link_with : libgalahad)
test('lsrt', lsrtt)
test('lsrti', lsrtti)

l2rtt = executable('l2rtt', 'src/l2rt/l2rtt.f90', link_with : libgalahad)
l2rtti = executable('l2rtti', 'src/l2rt/l2rtti.f90', link_with : libgalahad)
test('l2rt', l2rtt)
test('l2rti', l2rtti)

lhst = executable('lhst', 'src/lhs/lhst.f90', link_with : libgalahad)
test('lhs', lhst)

lmst = executable('lmst', 'src/lms/lmst.f90', link_with : libgalahad)
test('lms', lmst)

miqrt = executable('miqrt', 'src/miqr/miqrt.f90', link_with : libgalahad)
test('miqr', miqrt)

mopt = executable('mopt', 'src/mop/mopt.f90', link_with : [libgalahad_blas, libgalahad])
test('mop', mopt)

nlptt = executable('nlpt', 'src/nlpt/nlptt.f90', link_with : libgalahad)
test('nlpt', nlptt)

presolvet = executable('presolvet', 'src/presolve/presolvet.f90', link_with : libgalahad)
presolveti = executable('presolveti', 'src/presolve/presolveti.f90', link_with : libgalahad)
test('presolve', presolvet)
test('presolvei', presolveti)

qppt = executable('qppt', 'src/qpp/qppt.f90', link_with : libgalahad)
test('qpp', qppt)

qptt = executable('qptt', 'src/qpt/qptt.f90', link_with : libgalahad)
test('qpt', qptt)

randt = executable('randt', 'src/rand/randt.f90', link_with : libgalahad)
test('rand', randt)

rpdt = executable('rpdt', 'src/rpd/rpdt.f90', link_with : libgalahad)
rpdti = executable('rpdti', 'src/rpd/rpdti.f90', link_with : libgalahad)
# FIXME: install data files
# test('rpd', rpdt)
# test('rpdi', rpdti)

rootst = executable('rootst', 'src/roots/rootst.f90', link_with : libgalahad)
test('roots', rootst)

scut = executable('scut', 'src/scu/scut.f90', link_with : libgalahad)
test('scu', scut)

scalet = executable('scalet', 'src/scale/scalet.f90', link_with : libgalahad)
test('scale', scalet)

sect = executable('sect', 'src/sec/sect.f90', link_with : libgalahad)
test('sec', sect)

shat = executable('shat', 'src/sha/shat.f90', link_with : libgalahad)
test('sha', shat)

smtt = executable('smtt', 'src/smt/smtt.f90', link_with : libgalahad)
test('smt', smtt)

sortt = executable('sortt', 'src/sort/sortt.f90', link_with : libgalahad)
test('sort', sortt)

ugot = executable('ugot', 'src/ugo/ugot.f90', link_with : libgalahad)
# ugot2 = executable('ugot2', 'src/ugo/ugot2.f90', link_with : libgalahad)  # interactive
test('ugo', ugot)
# test('ugo2', ugot2)

if build_ciface
  # TODO: test C interface
  # gltr, glrt, lstr, lsrt, l2rt, presolve, rpd, ugo
endif
