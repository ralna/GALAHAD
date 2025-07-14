function run_sif(::Val{:dlp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.rundlp_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:dlp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.rundlp_sif_double()) $path_libsif $path_outsdif`)
end

function run_qplib(::Val{:dlp}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.rundlp_qplib_single())`; stdin=io)
    return run(process)
  end
end

function run_qplib(::Val{:dlp}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.rundlp_qplib_double())`; stdin=io)
    return run(process)
  end
end
