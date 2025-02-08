module GALAHAD

using Libdl
using LinearAlgebra
using Quadmath
import Quadmath.Cfloat128

if haskey(ENV, "JULIA_GALAHAD_LIBRARY_PATH")
  const libgalahad_single = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_single.$dlext")
  const libgalahad_double = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_double.$dlext")
  const libgalahad_quadruple = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_quadruple.$dlext")
  const libgalahad_single_64 = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_single_64.$dlext")
  const libgalahad_double_64 = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_double_64.$dlext")
  const libgalahad_quadruple_64 = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_quadruple_64.$dlext")
  const GALAHAD_INSTALLATION = "CUSTOM"
else
  import OpenBLAS32_jll
  import GALAHAD_jll
  const GALAHAD_INSTALLATION = "YGGDRASIL"
  const libgalahad_single = GALAHAD_jll.libgalahad_single
  const libgalahad_double = GALAHAD_jll.libgalahad_double
  const libgalahad_quadruple = replace(libgalahad_double, "double" => "quadruple")
  const libgalahad_single_64 = GALAHAD_jll.libgalahad_single_64
  const libgalahad_double_64 = GALAHAD_jll.libgalahad_double_64
  const libgalahad_quadruple_64 = replace(libgalahad_double_64, "double" => "quadruple")
end

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

# sls requires sils.
include("wrappers/sls.jl")

# rqs requires sls, sils, ir.
include("wrappers/rqs.jl")

# dps requires sls, sils.
include("wrappers/dps.jl")

# psls requires sls, sils.
include("wrappers/psls.jl")

# arc requires rqs, sls, sils, ir, glrt, dps, psls, lms, sha.
include("wrappers/arc.jl")

# trs requires sls, sils, ir.
include("wrappers/trs.jl")

# trb requires trs, sls, sils, ir, gltr, psls, lms, sha.
include("wrappers/trb.jl")

# bgo requires trb, trs, sls, sils, ir, gltr, psls, lms, sha, ugo, lhs.
include("wrappers/bgo.jl")

# uls requires gls.
include("wrappers/uls.jl")

# sbls requires sls, sils, uls, gls.
include("wrappers/sbls.jl")

# blls requires sbls, sls, sils, uls, gls, convert.
include("wrappers/blls.jl")

# bqp requires sbls, sls, sils, uls, gls.
include("wrappers/bqp.jl")

# fdc requires sls, sils, uls, gls.
include("wrappers/fdc.jl")

# cro requires sls, sils, sbls, uls, gls, ir, scu.
include("wrappers/cro.jl")

# bqpb requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/bqpb.jl")

# ccqp requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/ccqp.jl")

# cqp requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/cqp.jl")

# clls requires fdc, sls, sils, uls, gls, fit, roots, cro, ir, scu, rpd.
include("wrappers/clls.jl")

# dgo requires trb, trs, sls, sils, ir, gltr, psls, lms, sha, ugo, hash.
include("wrappers/dgo.jl")

# dqp requires fdc, sls, sils, uls, gls, sbls, gltr, scu, rpd.
include("wrappers/dqp.jl")

# eqp requires fdc, sls, sils, uls, gls, sbls, gltr.
include("wrappers/eqp.jl")

# lpa requires rpd.
include("wrappers/lpa.jl")

# lpb requires fdc, sls, sils, uls, gls, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/lpb.jl")

# lsqp requires fdc, sls, sils, uls, gls, sbls.
include("wrappers/lsqp.jl")

# nls requires rqs, sls, sils, ir, glrt, psls, bsc, roots.
include("wrappers/nls.jl")

# qpa requires sls, sils.
include("wrappers/qpa.jl")

# qpb requires lsqp, fdc, sls, sils, uls, gls, sbls, gltr, fit.
include("wrappers/qpb.jl")

# slls requires sbls, sls, sils, uls, gls, convert.
include("wrappers/slls.jl")

# tru requires trs, sls, sils, ir, gltr, dps, psls, lms, sec, sha.
include("wrappers/tru.jl")

# wcp requires fdc, sls, sils, uls, gls, sbls.
include("wrappers/wcp.jl")

# llsr requires sbls, sls, sils, uls, gls, ir.
include("wrappers/llsr.jl")

# llst requires sbls, sls, sils, uls, gls, ir.
include("wrappers/llst.jl")

# bllsb requires fdc, sbls, sls, sils, uls, gls, fit, roots, cro, rpd, ir.
include("wrappers/bllsb.jl")

# bnls requires rqs, sls, sils, ir, glrt, psls, bsc, roots.
include("wrappers/bnls.jl")

end # module GALAHAD
