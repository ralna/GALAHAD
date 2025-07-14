function run_sif(::Val{:expo}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runexpo_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:expo}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runexpo_sif_double()) $path_libsif $path_outsdif`)
end
