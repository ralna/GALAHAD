const runlqr_sif_single = joinpath(galahad_bindir, "runlqr_sif_single$(exeext)")

function run_sif(::Val{:lqr}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runlqr_sif_single $path_libsif $path_outsdif`)
end

const runlqr_sif_double = joinpath(galahad_bindir, "runlqr_sif_double$(exeext)")

function run_sif(::Val{:lqr}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runlqr_sif_double $path_libsif $path_outsdif`)
end

const runlqr_sif_quadruple = joinpath(galahad_bindir, "runlqr_sif_quadruple$(exeext)")

function run_sif(::Val{:lqr}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runlqr_sif_quadruple $path_libsif $path_outsdif`)
end
