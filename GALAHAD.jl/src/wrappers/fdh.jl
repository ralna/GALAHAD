const runfdh_sif_single = joinpath(galahad_bindir, "runfdh_sif_single$(exeext)")

function run_sif(::Val{:fdh}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runfdh_sif_single $path_libsif $path_outsdif`)
end

const runfdh_sif_double = joinpath(galahad_bindir, "runfdh_sif_double$(exeext)")

function run_sif(::Val{:fdh}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runfdh_sif_double $path_libsif $path_outsdif`)
end

const runfdh_sif_quadruple = joinpath(galahad_bindir, "runfdh_sif_quadruple$(exeext)")

function run_sif(::Val{:fdh}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runfdh_sif_quadruple $path_libsif $path_outsdif`)
end
