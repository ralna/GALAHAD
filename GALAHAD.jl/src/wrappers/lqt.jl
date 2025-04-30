const runlqt_sif_single = joinpath(galahad_bindir, "runlqt_sif_single$(exeext)")

function run_sif(::Val{:lqt}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runlqt_sif_single $path_libsif $path_outsdif`)
end

const runlqt_sif_double = joinpath(galahad_bindir, "runlqt_sif_double$(exeext)")

function run_sif(::Val{:lqt}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runlqt_sif_double $path_libsif $path_outsdif`)
end

const runlqt_sif_quadruple = joinpath(galahad_bindir, "runlqt_sif_quadruple$(exeext)")

function run_sif(::Val{:lqt}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runlqt_sif_quadruple $path_libsif $path_outsdif`)
end
