module GALAHAD

using Accessors
using Libdl
using LinearAlgebra
using Quadmath
import Quadmath.Cfloat128

if haskey(ENV, "JULIA_GALAHAD_LIBRARY_PATH")
  const galahad_libdir = ENV["JULIA_GALAHAD_LIBRARY_PATH"]
  const galahad_bindir = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "..", "bin") |> normpath
  const GALAHAD_INSTALLATION = "CUSTOM"
else
  import OpenBLAS32_jll
  import GALAHAD_jll
  const galahad_libdir = Sys.iswindows() ? joinpath(GALAHAD_jll.artifact_dir, "bin") : joinpath(GALAHAD_jll.artifact_dir, "lib")
  const galahad_bindir = joinpath(GALAHAD_jll.artifact_dir, "bin")
  const GALAHAD_INSTALLATION = "YGGDRASIL"
end
const exeext = Sys.iswindows() ? ".exe" : ""

# Shared libraries of GALAHAD
const libgalahad_single = joinpath(galahad_libdir, "libgalahad_single.$dlext")
const libgalahad_double = joinpath(galahad_libdir, "libgalahad_double.$dlext")
const libgalahad_quadruple = joinpath(galahad_libdir, "libgalahad_quadruple.$dlext")
const libgalahad_single_64 = joinpath(galahad_libdir, "libgalahad_single_64.$dlext")
const libgalahad_double_64 = joinpath(galahad_libdir, "libgalahad_double_64.$dlext")
const libgalahad_quadruple_64 = joinpath(galahad_libdir, "libgalahad_quadruple_64.$dlext")

function __init__()
  if GALAHAD_INSTALLATION == "YGGDRASIL"
    config = LinearAlgebra.BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
      LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
  end
end

# Utils.
include("utils.jl")

# packages without dependencies.
include("wrappers/bsc.jl")
include("wrappers/convert.jl")
include("wrappers/fit.jl")
include("wrappers/glrt.jl")
include("wrappers/gls.jl")
include("wrappers/gltr.jl")
include("wrappers/hash.jl")
include("wrappers/hsl.jl")
include("wrappers/ir.jl")
include("wrappers/l2rt.jl")
include("wrappers/lhs.jl")
include("wrappers/lms.jl")
include("wrappers/lsrt.jl")
include("wrappers/lstr.jl")
include("wrappers/nodend.jl")
include("wrappers/presolve.jl")
include("wrappers/roots.jl")
include("wrappers/rpd.jl")
include("wrappers/scu.jl")
include("wrappers/sec.jl")
include("wrappers/sha.jl")
include("wrappers/sils.jl")
include("wrappers/ugo.jl")
include("wrappers/ssids.jl")
include("wrappers/version.jl")

# packages without a C interface -- only binaries run_sif.
include("wrappers/cdqp.jl")
include("wrappers/demo.jl")
include("wrappers/dlp.jl")
include("wrappers/fdh.jl")
include("wrappers/filtrane.jl")
include("wrappers/l1qp.jl")
include("wrappers/lancelot.jl")
include("wrappers/lls.jl")
include("wrappers/lpqp.jl")
include("wrappers/lqr.jl")
include("wrappers/lqt.jl")
include("wrappers/miqr.jl")
include("wrappers/qp.jl")
include("wrappers/qpc.jl")
include("wrappers/warm.jl")

# sls requires sils, nodend.
include("wrappers/sls.jl")

# rqs requires sls, sils, ir, nodend.
include("wrappers/rqs.jl")

# dps requires sls, sils, nodend.
include("wrappers/dps.jl")

# psls requires sls, sils, nodend.
include("wrappers/psls.jl")

# arc requires rqs, sls, sils, ir, glrt, dps, psls, lms, sha, nodend.
include("wrappers/arc.jl")

# trs requires sls, sils, ir, nodend.
include("wrappers/trs.jl")

# trb requires trs, sls, sils, ir, gltr, psls, lms, sha, nodend.
include("wrappers/trb.jl")

# bgo requires trb, trs, sls, sils, ir, gltr, psls, lms, sha, ugo, lhs, nodend.
include("wrappers/bgo.jl")

# uls requires gls.
include("wrappers/uls.jl")

# sbls requires sls, sils, uls, gls, nodend.
include("wrappers/sbls.jl")

# blls requires sbls, sls, sils, uls, gls, convert, nodend.
include("wrappers/blls.jl")

# bqp requires sbls, sls, sils, uls, gls, nodend.
include("wrappers/bqp.jl")

# fdc requires sls, sils, uls, gls, nodend.
include("wrappers/fdc.jl")

# cro requires sls, sils, sbls, uls, gls, ir, scu, nodend.
include("wrappers/cro.jl")

# bqpb requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd, nodend.
include("wrappers/bqpb.jl")

# ccqp requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/ccqp.jl")

# cqp requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd, nodend.
include("wrappers/cqp.jl")

# clls requires fdc, sls, sils, uls, gls, fit, roots, cro, ir, scu, rpd, nodend.
include("wrappers/clls.jl")

# dgo requires trb, trs, sls, sils, ir, gltr, psls, lms, sha, ugo, hash, nodend.
include("wrappers/dgo.jl")

# dqp requires fdc, sls, sils, uls, gls, sbls, gltr, scu, rpd, nodend.
include("wrappers/dqp.jl")

# eqp requires fdc, sls, sils, uls, gls, sbls, gltr, nodend.
include("wrappers/eqp.jl")

# lpa requires rpd.
include("wrappers/lpa.jl")

# lpb requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd, nodend.
include("wrappers/lpb.jl")

# lsqp requires fdc, sls, sils, uls, gls, sbls, nodend.
include("wrappers/lsqp.jl")

# nls requires rqs, sls, sils, ir, glrt, psls, bsc, roots, nodend.
include("wrappers/nls.jl")

# qpa requires sls, sils, nodend.
include("wrappers/qpa.jl")

# qpb requires lsqp, fdc, sls, sils, uls, gls, sbls, gltr, fit, nodend.
include("wrappers/qpb.jl")

# slls requires sbls, sls, sils, uls, gls, convert, nodend.
include("wrappers/slls.jl")

# tru requires trs, sls, sils, ir, gltr, dps, psls, lms, sec, sha, nodend.
include("wrappers/tru.jl")

# wcp requires fdc, sls, sils, uls, gls, sbls, nodend.
include("wrappers/wcp.jl")

# llsr requires sbls, sls, sils, uls, gls, ir, nodend.
include("wrappers/llsr.jl")

# llst requires sbls, sls, sils, uls, gls, ir, nodend.
include("wrappers/llst.jl")

# bllsb requires fdc, sbls, sls, sils, uls, gls, fit, roots, cro, rpd, ir, nodend.
include("wrappers/bllsb.jl")

# bnls requires rqs, sls, sils, ir, glrt, psls, bsc, roots, nodend.
include("wrappers/bnls.jl")

# ssls requires sls, sils, nodend.
include("wrappers/ssls.jl")

# expo requires bsc, tru, ssls, sls, sils, trs, ir, gltr, dps, psls, lms, sec, sha, nodend.
include("wrappers/expo.jl")

end # module GALAHAD
