const runcdqp_sif_single = joinpath(galahad_bindir, "runcdqp_sif_single$(exeext)")

function run_sif(::Val{:cdqp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runcdqp_sif_single $path_libsif $path_outsdif`)
end

const runcdqp_sif_double = joinpath(galahad_bindir, "runcdqp_sif_double$(exeext)")

function run_sif(::Val{:cdqp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runcdqp_sif_double $path_libsif $path_outsdif`)
end

const runcdqp_sif_quadruple = joinpath(galahad_bindir, "runcdqp_sif_quadruple$(exeext)")

function run_sif(::Val{:cdqp}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runcdqp_sif_quadruple $path_libsif $path_outsdif`)
end

const runcdqp_qplib_single = joinpath(galahad_bindir, "runcdqp_qplib_single$(exeext)")

function run_qplib(::Val{:cdqp}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runcdqp_qplib_single`; stdin=io)
  end
end

const runcdqp_qplib_double = joinpath(galahad_bindir, "runcdqp_qplib_double$(exeext)")

function run_qplib(::Val{:cdqp}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runcdqp_qplib_double`; stdin=io)
  end
end

const runcdqp_qplib_quadruple = joinpath(galahad_bindir, "runcdqp_qplib_quadruple$(exeext)")

function run_qplib(::Val{:cdqp}, ::Val{:quadruple}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runcdqp_qplib_quadruple`; stdin=io)
  end
end
