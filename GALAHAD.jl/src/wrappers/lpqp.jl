function run_sif(::Val{:lpqp}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlpqp_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:lpqp}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlpqp_sif_double()) $path_libsif $path_outsdif`)
end
