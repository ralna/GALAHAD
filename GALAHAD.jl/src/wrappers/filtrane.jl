function run_sif(::Val{:filtrane}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  return run(`$(GALAHAD_jll.runfiltrane_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:filtrane}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  return run(`$(GALAHAD_jll.runfiltrane_sif_double()) $path_libsif $path_outsdif`)
end
