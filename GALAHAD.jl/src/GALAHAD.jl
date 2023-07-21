module GALAHAD

using Libdl

if haskey(ENV, "JULIA_GALAHAD_LIBRARY_PATH")
  const libgalahad_single = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_single.$dlext")
  const libgalahad_double = joinpath(ENV["JULIA_GALAHAD_LIBRARY_PATH"], "libgalahad_double.$dlext")
  const GALAHAD_INSTALLATION = "CUSTOM"
else
  # using GALAHAD_jll
  # const GALAHAD_INSTALLATION = "YGGDRASIL"
end

galahad_version() = v"5.0.0"

# packages without dependencies.
include("wrappers/bsc.jl")
include("wrappers/convert.jl")
include("wrappers/fit.jl")
include("wrappers/glrt.jl")
include("wrappers/gls.jl")
include("wrappers/gltr.jl")
include("wrappers/hash.jl")
include("wrappers/hsl.jl")
include("wrappers/icfs.jl")
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

# sls requires sils, ma57, ma77, ma86, ma87, ma97, ssids, mc64, mc68.
include("wrappers/sls.jl")

# rqs requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir.
include("wrappers/rqs.jl")

# dps requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids, mc64, mc68.
include("wrappers/dps.jl")

# psls requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, mi28.
include("wrappers/psls.jl")

# arc requires rqs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, glrt, dps, psls, mi28, lms, sha.
include("wrappers/arc.jl")

# trs requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir.
include("wrappers/trs.jl")

# trb requires trs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, gltr, psls, mi28, lms, sha.
include("wrappers/trb.jl")

# bgo requires trb, trs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, gltr, psls, mi28, lms, sha, ugo, lhs.
include("wrappers/bgo.jl")

# uls requires gls, ma48.
include("wrappers/uls.jl")

# sbls requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48.
include("wrappers/sbls.jl")

# blls requires sbls, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, convert.
include("wrappers/blls.jl")

# bqp requires sbls, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48.
include("wrappers/bqp.jl")

# fdc requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48.
include("wrappers/fdc.jl")

# cro requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, sbls, uls, gls, ma48, ir, scu.
include("wrappers/cro.jl")

# bqpb requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/bqpb.jl")

# ccqp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/ccqp.jl")

# cqp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/cqp.jl")

# dgo requires trb, trs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, gltr, psls, mi28, lms, sha, ugo, hash.
include("wrappers/dgo.jl")

# dqp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, gltr, scu, rpd.
include("wrappers/dqp.jl")

# eqp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, gltr.
include("wrappers/eqp.jl")

# lpa requires rpd.
include("wrappers/lpa.jl")

# lpb requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, fit, roots, cro, ir, scu, rpd.
include("wrappers/lpb.jl")

# lsqp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls.
include("wrappers/lsqp.jl")

# nls requires rqs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, glrt, psls, mi28, bsc, roots.
include("wrappers/nls.jl")

# qpa requires sls, sils, ma57, ma77, ma86, ma87, ma97, ssids, mc64, mc68.
include("wrappers/qpa.jl")

# qpb requires lsqp, fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls, gltr, fit.
include("wrappers/qpb.jl")

# slls requires sbls, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, convert.
include("wrappers/slls.jl")

# tru requires trs, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, ir, gltr, dps, psls, mi28, lms, sec, sha.
include("wrappers/tru.jl")

# wcp requires fdc, sls, sils, ma57, ma77, ma86, ma87, ma97, ssids,
# mc64, mc68, uls, gls, ma48, sbls.
include("wrappers/wcp.jl")

end # module GALAHAD
