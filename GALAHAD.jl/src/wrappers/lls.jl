function run_sif(::Val{:lls}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlls_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:lls}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runlls_sif_double()) $path_libsif $path_outsdif`)
end
