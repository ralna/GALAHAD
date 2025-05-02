const runqpc_sif_single = joinpath(galahad_bindir, "runqpc_sif_single$(exeext)")

function run_sif(::Val{:qpc}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runqpc_sif_single $path_libsif $path_outsdif`)
end

const runqpc_sif_double = joinpath(galahad_bindir, "runqpc_sif_double$(exeext)")

function run_sif(::Val{:qpc}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runqpc_sif_double $path_libsif $path_outsdif`)
end

const runqpc_sif_quadruple = joinpath(galahad_bindir, "runqpc_sif_quadruple$(exeext)")

function run_sif(::Val{:qpc}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runqpc_sif_quadruple $path_libsif $path_outsdif`)
end

const runqpc_qplib_single = joinpath(galahad_bindir, "runqpc_qplib_single$(exeext)")

function run_qplib(::Val{:qpc}, ::Val{:single}, path_qplib::String)
  return run(`$runqpc_qplib_single $path_qplib`)
end

const runqpc_qplib_double = joinpath(galahad_bindir, "runqpc_qplib_double$(exeext)")

function run_qplib(::Val{:qpc}, ::Val{:double}, path_qplib::String)
  return run(`$runqpc_qplib_double $path_qplib`)
end

const runqpc_qplib_quadruple = joinpath(galahad_bindir, "runqpc_qplib_quadruple$(exeext)")

function run_qplib(::Val{:qpc}, ::Val{:quadruple}, path_qplib::String)
  return run(`$runqpc_qplib_quadruple $path_qplib`)
end
