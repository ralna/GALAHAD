function run_sif(::Val{:qp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runqp_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:qp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runqp_sif_double()) $path_libsif $path_outsdif`)
end

function run_qplib(::Val{:qp}, ::Val{:single}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.runqp_qplib_single())`; stdin=io)
    return run(process)
  end
end

function run_qplib(::Val{:qp}, ::Val{:double}, path_qplib::String)
  open(path_qplib, "r") do io
    process = pipeline(`$(GALAHAD_jll.runqp_qplib_double())`; stdin=io)
    return run(process)
  end
end
