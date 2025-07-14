function run_sif(::Val{:warm}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runwarm_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:warm}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runwarm_sif_double()) $path_libsif $path_outsdif`)
end
