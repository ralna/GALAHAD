function run_sif(::Val{:lancelot}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  return run(`$(GALAHAD_jll.runlancelot_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:lancelot}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  return run(`$(GALAHAD_jll.runlancelot_sif_double()) $path_libsif $path_outsdif`)
end
