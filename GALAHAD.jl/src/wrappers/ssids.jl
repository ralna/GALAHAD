export spral_ssids_options

struct spral_ssids_options{T,INT}
  array_base::INT
  print_level::INT
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  ordering::INT
  nemin::INT
  ignore_numa::Bool
  use_gpu::Bool
  gpu_only::Bool
  min_gpu_work::Int64
  max_load_inbalance::Float32
  gpu_perf_coeff::Float32
  scaling::INT
  small_subtree_threshold::Int64
  cpu_block_size::INT
  action::Bool
  pivot_method::INT
  small::T
  u::T
  nodend_options::nodend_control_type{INT}
  nstream::INT
  multiplier::T
  min_loadbalance::Float32
  failed_pivot_method::INT
end

export spral_ssids_inform

struct spral_ssids_inform{T,INT}
  flag::INT
  matrix_dup::INT
  matrix_missing_diag::INT
  matrix_outrange::INT
  matrix_rank::INT
  maxdepth::INT
  maxfront::INT
  maxsupernode::INT
  num_delay::INT
  num_factor::Int64
  num_flops::Int64
  num_neg::INT
  num_sup::INT
  num_two::INT
  stat::INT
  cuda_error::INT
  cublas_error::INT
  nodend_inform::nodend_inform_type{T,INT}
  not_first_pass::INT
  not_second_pass::INT
  nparts::INT
  cpu_flops::Int64
  gpu_flops::Int64
end

function run_sif(::Val{:ssids}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runssids_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:ssids}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runssids_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
