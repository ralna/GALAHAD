function run_sif(::Val{:l1qp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runl1qp_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:l1qp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runl1qp_sif_double()) $path_libsif $path_outsdif`)
end
