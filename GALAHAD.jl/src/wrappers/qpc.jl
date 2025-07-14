function run_sif(::Val{:qpc}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runqpc_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:qpc}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runqpc_sif_double()) $path_libsif $path_outsdif`)
end

function run_qplib(::Val{:qpc}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.runqpc_qplib_single())`; stdin=io)
    return run(process)
  end
end

function run_qplib(::Val{:qpc}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.runqpc_qplib_double())`; stdin=io)
    return run(process)
  end
end
