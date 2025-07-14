function run_sif(::Val{:fdh}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runfdh_sif_single()) $path_libsif $path_outsdif`)
end

function run_sif(::Val{:fdh}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  return run(`$(GALAHAD_jll.runfdh_sif_double()) $path_libsif $path_outsdif`)
end
