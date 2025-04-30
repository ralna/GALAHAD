const runwarm_sif_single = joinpath(galahad_bindir, "runwarm_sif_single$(exeext)")

function run_sif(::Val{:warm}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runwarm_sif_single $path_libsif $path_outsdif`)
end

const runwarm_sif_double = joinpath(galahad_bindir, "runwarm_sif_double$(exeext)")

function run_sif(::Val{:warm}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runwarm_sif_double $path_libsif $path_outsdif`)
end

const runwarm_sif_quadruple = joinpath(galahad_bindir, "runwarm_sif_quadruple$(exeext)")

function run_sif(::Val{:warm}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runwarm_sif_quadruple $path_libsif $path_outsdif`)
end
