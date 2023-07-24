export sls_control_type

mutable struct sls_control_type{T}
  f_indexing::Bool
  error::Cint
  warning::Cint
  out::Cint
  statistics::Cint
  print_level::Cint
  print_level_solver::Cint
  bits::Cint
  block_size_kernel::Cint
  block_size_elimination::Cint
  blas_block_size_factorize::Cint
  blas_block_size_solve::Cint
  node_amalgamation::Cint
  initial_pool_size::Cint
  min_real_factor_size::Cint
  min_integer_factor_size::Cint
  max_real_factor_size::Int64
  max_integer_factor_size::Int64
  max_in_core_store::Int64
  array_increase_factor::T
  array_decrease_factor::T
  pivot_control::Cint
  ordering::Cint
  full_row_threshold::Cint
  row_search_indefinite::Cint
  scaling::Cint
  scale_maxit::Cint
  scale_thresh::T
  relative_pivot_tolerance::T
  minimum_pivot_tolerance::T
  absolute_pivot_tolerance::T
  zero_tolerance::T
  zero_pivot_tolerance::T
  negative_pivot_tolerance::T
  static_pivot_tolerance::T
  static_level_switch::T
  consistency_tolerance::T
  max_iterative_refinements::Cint
  acceptable_residual_relative::T
  acceptable_residual_absolute::T
  multiple_rhs::Bool
  generate_matrix_file::Bool
  matrix_file_device::Cint
  matrix_file_name::NTuple{31,Cchar}
  out_of_core_directory::NTuple{401,Cchar}
  out_of_core_integer_factor_file::NTuple{401,Cchar}
  out_of_core_real_factor_file::NTuple{401,Cchar}
  out_of_core_real_work_file::NTuple{401,Cchar}
  out_of_core_indefinite_file::NTuple{401,Cchar}
  out_of_core_restart_file::NTuple{501,Cchar}
  prefix::NTuple{31,Cchar}

  sls_control_type{T}() where T = new()
end

export sls_time_type

mutable struct sls_time_type{T}
  total::T
  analyse::T
  factorize::T
  solve::T
  order_external::T
  analyse_external::T
  factorize_external::T
  solve_external::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
  clock_order_external::T
  clock_analyse_external::T
  clock_factorize_external::T
  clock_solve_external::T

  sls_time_type{T}() where T = new()
end

export sls_inform_type

mutable struct sls_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  more_info::Cint
  entries::Cint
  out_of_range::Cint
  duplicates::Cint
  upper::Cint
  missing_diagonals::Cint
  max_depth_assembly_tree::Cint
  nodes_assembly_tree::Cint
  real_size_desirable::Int64
  integer_size_desirable::Int64
  real_size_necessary::Int64
  integer_size_necessary::Int64
  real_size_factors::Int64
  integer_size_factors::Int64
  entries_in_factors::Int64
  max_task_pool_size::Cint
  max_front_size::Cint
  compresses_real::Cint
  compresses_integer::Cint
  two_by_two_pivots::Cint
  semi_bandwidth::Cint
  delayed_pivots::Cint
  pivot_sign_changes::Cint
  static_pivots::Cint
  first_modified_pivot::Cint
  rank::Cint
  negative_eigenvalues::Cint
  num_zero::Cint
  iterative_refinements::Cint
  flops_assembly::Int64
  flops_elimination::Int64
  flops_blas::Int64
  largest_modified_pivot::T
  minimum_scaling_factor::T
  maximum_scaling_factor::T
  condition_number_1::T
  condition_number_2::T
  backward_error_1::T
  backward_error_2::T
  forward_error::T
  alternative::Bool
  solver::NTuple{21,Cchar}
  time::sls_time_type{T}
  sils_ainfo::sils_ainfo_type{T}
  sils_finfo::sils_finfo_type{T}
  sils_sinfo::sils_sinfo_type{T}
  ma57_ainfo::ma57_ainfo{T}
  ma57_finfo::ma57_finfo{T}
  ma57_sinfo::ma57_sinfo{T}
  ma77_info::ma77_info{T}
  ma86_info::ma86_info{T}
  ma87_info::ma87_info{T}
  ma97_info::ma97_info{T}
  ssids_inform::spral_ssids_inform
  mc61_info::NTuple{10,Cint}
  mc61_rinfo::NTuple{15,T}
  mc64_info::mc64_info
  mc68_info::mc68_info
  mc77_info::NTuple{10,Cint}
  mc77_rinfo::NTuple{10,T}
  mumps_error::Cint
  mumps_info::NTuple{80,Cint}
  mumps_rinfo::NTuple{40,T}
  pardiso_error::Cint
  pardiso_IPARM::NTuple{64,Cint}
  pardiso_DPARM::NTuple{64,T}
  mkl_pardiso_error::Cint
  mkl_pardiso_IPARM::NTuple{64,Cint}
  pastix_info::Cint
  wsmp_error::Cint
  wsmp_iparm::NTuple{64,Cint}
  wsmp_dparm::NTuple{64,T}
  mpi_ierr::Cint
  lapack_error::Cint

  function sls_inform_type{T}() where T
    type = new()
    type.time = sls_time_type{T}()
    type.sils_ainfo = sils_ainfo_type{T}()
    type.sils_finfo = sils_finfo_type{T}()
    type.sils_sinfo = sils_sinfo_type{T}()
    type.ma57_ainfo = ma57_ainfo{T}()
    type.ma57_finfo = ma57_finfo{T}()
    type.ma57_sinfo = ma57_sinfo{T}()
    type.ma77_info = ma77_info{T}()
    type.ma86_info = ma86_info{T}()
    type.ma87_info = ma87_info{T}()
    type.ma97_info = ma97_info{T}()
    type.ssids_inform = spral_ssids_inform()
    type.mc64_info = mc64_info()
    type.mc68_info = mc68_info()
    return type
  end
end

export sls_initialize_s

function sls_initialize_s(solver, data, control, status)
  @ccall libgalahad_single.sls_initialize_s(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                            control::Ref{sls_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export sls_initialize

function sls_initialize(solver, data, control, status)
  @ccall libgalahad_double.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ref{sls_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export sls_read_specfile_s

function sls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.sls_read_specfile_s(control::Ref{sls_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export sls_read_specfile

function sls_read_specfile(control, specfile)
  @ccall libgalahad_double.sls_read_specfile(control::Ref{sls_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export sls_analyse_matrix_s

function sls_analyse_matrix_s(control, data, status, n, type, ne, row, col, ptr)
  @ccall libgalahad_single.sls_analyse_matrix_s(control::Ref{sls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, type::Ptr{Cchar}, ne::Cint,
                                                row::Ptr{Cint}, col::Ptr{Cint},
                                                ptr::Ptr{Cint})::Cvoid
end

export sls_analyse_matrix

function sls_analyse_matrix(control, data, status, n, type, ne, row, col, ptr)
  @ccall libgalahad_double.sls_analyse_matrix(control::Ref{sls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, type::Ptr{Cchar}, ne::Cint,
                                              row::Ptr{Cint}, col::Ptr{Cint},
                                              ptr::Ptr{Cint})::Cvoid
end

export sls_reset_control_s

function sls_reset_control_s(control, data, status)
  @ccall libgalahad_single.sls_reset_control_s(control::Ref{sls_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export sls_reset_control

function sls_reset_control(control, data, status)
  @ccall libgalahad_double.sls_reset_control(control::Ref{sls_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export sls_factorize_matrix_s

function sls_factorize_matrix_s(data, status, ne, val)
  @ccall libgalahad_single.sls_factorize_matrix_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  ne::Cint, val::Ptr{Float32})::Cvoid
end

export sls_factorize_matrix

function sls_factorize_matrix(data, status, ne, val)
  @ccall libgalahad_double.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                ne::Cint, val::Ptr{Float64})::Cvoid
end

export sls_solve_system_s

function sls_solve_system_s(data, status, n, sol)
  @ccall libgalahad_single.sls_solve_system_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, sol::Ptr{Float32})::Cvoid
end

export sls_solve_system

function sls_solve_system(data, status, n, sol)
  @ccall libgalahad_double.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, sol::Ptr{Float64})::Cvoid
end

export sls_partial_solve_system_s

function sls_partial_solve_system_s(part, data, status, n, sol)
  @ccall libgalahad_single.sls_partial_solve_system_s(part::Ptr{Cchar},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      sol::Ptr{Float32})::Cvoid
end

export sls_partial_solve_system

function sls_partial_solve_system(part, data, status, n, sol)
  @ccall libgalahad_double.sls_partial_solve_system(part::Ptr{Cchar},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    sol::Ptr{Float64})::Cvoid
end

export sls_information_s

function sls_information_s(data, inform, status)
  @ccall libgalahad_single.sls_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{sls_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export sls_information

function sls_information(data, inform, status)
  @ccall libgalahad_double.sls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{sls_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export sls_terminate_s

function sls_terminate_s(data, control, inform)
  @ccall libgalahad_single.sls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sls_control_type{Float32}},
                                           inform::Ref{sls_inform_type{Float32}})::Cvoid
end

export sls_terminate

function sls_terminate(data, control, inform)
  @ccall libgalahad_double.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{sls_control_type{Float64}},
                                         inform::Ref{sls_inform_type{Float64}})::Cvoid
end
