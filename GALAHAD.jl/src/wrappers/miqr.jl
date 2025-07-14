function run_sif(::Val{:miqr}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runmiqr_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:miqr}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runmiqr_sif_double()) $path_libsif $path_outsdif`)
end
