function run_sif(::Val{:qpc}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runqpc_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:qpc}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runqpc_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_qplib(::Val{:qpc}, ::Val{:single}, path_qplib::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runqpc_qplib_single())`)
  open(path_qplib, "r") do io
    process = pipeline(cmd; stdin=io)
    return run(process)
  end
  return nothing
end

function run_qplib(::Val{:qpc}, ::Val{:double}, path_qplib::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runqpc_qplib_double())`)
  open(path_qplib, "r") do io
    process = pipeline(cmd; stdin=io)
    return run(process)
  end
  return nothing
end
