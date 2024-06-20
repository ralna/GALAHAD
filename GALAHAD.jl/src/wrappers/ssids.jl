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
  min_gpu_work::Clong
  max_load_inbalance::Cfloat
  gpu_perf_coeff::Cfloat
  scaling::Cint
  small_subtree_threshold::Clong
  cpu_block_size::Cint
  action::Bool
  pivot_method::Cint
  small::T
  u::T
  nstream::Cint
  multiplier::T
  min_loadbalance::Cfloat
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
  num_factor::Clong
  num_flops::Clong
  num_neg::Cint
  num_sup::Cint
  num_two::Cint
  stat::Cint
  cuda_error::Cint
  cublas_error::Cint
  not_first_pass::Cint
  not_second_pass::Cint
  nparts::Cint
  cpu_flops::Clong
  gpu_flops::Clong
end
