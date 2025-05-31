const rundlp_sif_single = joinpath(galahad_bindir, "rundlp_sif_single$(exeext)")

function run_sif(::Val{:dlp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$rundlp_sif_single $path_libsif $path_outsdif`)
end

const rundlp_sif_double = joinpath(galahad_bindir, "rundlp_sif_double$(exeext)")

function run_sif(::Val{:dlp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$rundlp_sif_double $path_libsif $path_outsdif`)
end

const rundlp_sif_quadruple = joinpath(galahad_bindir, "rundlp_sif_quadruple$(exeext)")

function run_sif(::Val{:dlp}, ::Val{:quadruple}, path_libsif::String, path_outsdif::String)
  return run(`$rundlp_sif_quadruple $path_libsif $path_outsdif`)
end

const rundlp_qplib_single = joinpath(galahad_bindir, "rundlp_qplib_single$(exeext)")

function run_qplib(::Val{:dlp}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$rundlp_qplib_single`; stdin=io)
    return run(process)
  end
end

const rundlp_qplib_double = joinpath(galahad_bindir, "rundlp_qplib_double$(exeext)")

function run_qplib(::Val{:dlp}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$rundlp_qplib_double`; stdin=io)
    return run(process)
  end
end

const rundlp_qplib_quadruple = joinpath(galahad_bindir, "rundlp_qplib_quadruple$(exeext)")

function run_qplib(::Val{:dlp}, ::Val{:quadruple}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$rundlp_qplib_quadruple`; stdin=io)
    return run(process)
  end
end
