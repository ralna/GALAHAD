const runlls_sif_single = joinpath(galahad_bindir, "runlls_sif_single$(exeext)")

function run_sif(::Val{:lls}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runlls_sif_single $path_libsif $path_outsdif`)
end

const runlls_sif_double = joinpath(galahad_bindir, "runlls_sif_double$(exeext)")

function run_sif(::Val{:lls}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runlls_sif_double $path_libsif $path_outsdif`)
end

const runlls_sif_quadruple = joinpath(galahad_bindir, "runlls_sif_quadruple$(exeext)")

function run_sif(::Val{:lls}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runlls_sif_quadruple $path_libsif $path_outsdif`)
end
