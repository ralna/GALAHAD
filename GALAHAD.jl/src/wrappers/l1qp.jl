const runl1qp_sif_single = joinpath(galahad_bindir, "runl1qp_sif_single$(exeext)")

function run_sif(::Val{:l1qp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runl1qp_sif_single $path_libsif $path_outsdif`)
end

const runl1qp_sif_double = joinpath(galahad_bindir, "runl1qp_sif_double$(exeext)")

function run_sif(::Val{:l1qp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runl1qp_sif_double $path_libsif $path_outsdif`)
end

const runl1qp_sif_quadruple = joinpath(galahad_bindir, "runl1qp_sif_quadruple$(exeext)")

function run_sif(::Val{:l1qp}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runl1qp_sif_quadruple $path_libsif $path_outsdif`)
end
