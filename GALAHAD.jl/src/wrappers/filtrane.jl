const runfiltrane_sif_single = joinpath(galahad_bindir, "runfiltrane_sif_single$(exeext)")

function run_sif(::Val{:filtrane}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  return run(`$runfiltrane_sif_single $path_libsif $path_outsdif`)
end

const runfiltrane_sif_double = joinpath(galahad_bindir, "runfiltrane_sif_double$(exeext)")

function run_sif(::Val{:filtrane}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  return run(`$runfiltrane_sif_double $path_libsif $path_outsdif`)
end

const runfiltrane_sif_quadruple = joinpath(galahad_bindir,
                                           "runfiltrane_sif_quadruple$(exeext)")

function run_sif(::Val{:filtrane}, ::Val{:quadruple}, path_libsif::String,
                 path_outsdif::String)
  return run(`$runfiltrane_sif_quadruple $path_libsif $path_outsdif`)
end
