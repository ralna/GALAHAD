function run_sif(::Val{:demo}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.rundemo_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:demo}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.rundemo_sif_double()) $path_libsif $path_outsdif`)
end
