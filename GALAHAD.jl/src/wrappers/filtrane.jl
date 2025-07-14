function run_sif(::Val{:filtrane}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runfiltrane_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:filtrane}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runfiltrane_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
