const runqp_sif_single = joinpath(galahad_bindir, "runqp_sif_single$(exeext)")

function run_sif(::Val{:qp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$runqp_sif_single $path_libsif $path_outsdif`)
end

const runqp_sif_double = joinpath(galahad_bindir, "runqp_sif_double$(exeext)")

function run_sif(::Val{:qp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$runqp_sif_double $path_libsif $path_outsdif`)
end

const runqp_sif_quadruple = joinpath(galahad_bindir, "runqp_sif_quadruple$(exeext)")

function run_sif(::Val{:qp}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$runqp_sif_quadruple $path_libsif $path_outsdif`)
end

const runqp_qplib_single = joinpath(galahad_bindir, "runqp_qplib_single$(exeext)")

function run_qplib(::Val{:qp}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runqp_qplib_single`; stdin=io)
  end
end

const runqp_qplib_double = joinpath(galahad_bindir, "runqp_qplib_double$(exeext)")

function run_qplib(::Val{:qp}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runqp_qplib_double`; stdin=io)
  end
end

const runqp_qplib_quadruple = joinpath(galahad_bindir, "runqp_qplib_quadruple$(exeext)")

function run_qplib(::Val{:qp}, ::Val{:quadruple}, path_qplib::String)
  open(path_qplib, "r") do io
    return run(`$runqp_qplib_quadruple`; stdin=io)
  end
end
