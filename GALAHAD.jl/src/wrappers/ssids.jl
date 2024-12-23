export spral_ssids_options

struct spral_ssids_options{T}
  array_base::Cint
  print_level::Cint
  unit_diagnostics::Cint
  unit_error::Cint
  unit_warning::Cint
  ordering::Cint
  nemin::Cint
  ignore_numa::Bool
  use_gpu::Bool
  gpu_only::Bool
  min_gpu_work::Int64
  max_load_inbalance::Float32
  gpu_perf_coeff::Float32
  scaling::Cint
  small_subtree_threshold::Int64
  cpu_block_size::Cint
  action::Bool
  pivot_method::Cint
  small::T
  u::T
  nstream::Cint
  multiplier::T
  min_loadbalance::Float32
  failed_pivot_method::Cint
end

export spral_ssids_inform

struct spral_ssids_inform
  flag::Cint
  matrix_dup::Cint
  matrix_missing_diag::Cint
  matrix_outrange::Cint
  matrix_rank::Cint
  maxdepth::Cint
  maxfront::Cint
  maxsupernode::Cint
  num_delay::Cint
  num_factor::Int64
  num_flops::Int64
  num_neg::Cint
  num_sup::Cint
  num_two::Cint
  stat::Cint
  cuda_error::Cint
  cublas_error::Cint
  not_first_pass::Cint
  not_second_pass::Cint
  nparts::Cint
  cpu_flops::Int64
  gpu_flops::Int64
end
