function run_sif(::Val{:lqt}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlqt_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:lqt}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlqt_sif_double()) $path_libsif $path_outsdif`)
end
