export sls_control_type

struct sls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  warning::INT
  out::INT
  statistics::INT
  print_level::INT
  print_level_solver::INT
  bits::INT
  block_size_kernel::INT
  block_size_elimination::INT
  blas_block_size_factorize::INT
  blas_block_size_solve::INT
  node_amalgamation::INT
  initial_pool_size::INT
  min_real_factor_size::INT
  min_integer_factor_size::INT
  max_real_factor_size::Int64
  max_integer_factor_size::Int64
  max_in_core_store::Int64
  array_increase_factor::T
  array_decrease_factor::T
  pivot_control::INT
  ordering::INT
  full_row_threshold::INT
  row_search_indefinite::INT
  scaling::INT
  scale_maxit::INT
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
  max_iterative_refinements::INT
  acceptable_residual_relative::T
  acceptable_residual_absolute::T
  multiple_rhs::Bool
  generate_matrix_file::Bool
  matrix_file_device::INT
  matrix_file_name::NTuple{31,Cchar}
  out_of_core_directory::NTuple{401,Cchar}
  out_of_core_integer_factor_file::NTuple{401,Cchar}
  out_of_core_real_factor_file::NTuple{401,Cchar}
  out_of_core_real_work_file::NTuple{401,Cchar}
  out_of_core_indefinite_file::NTuple{401,Cchar}
  out_of_core_restart_file::NTuple{501,Cchar}
  prefix::NTuple{31,Cchar}
end

export sls_time_type

struct sls_time_type{T}
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
end

export sls_inform_type

struct sls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  more_info::INT
  entries::INT
  out_of_range::INT
  duplicates::INT
  upper::INT
  missing_diagonals::INT
  max_depth_assembly_tree::INT
  nodes_assembly_tree::INT
  real_size_desirable::Int64
  integer_size_desirable::Int64
  real_size_necessary::Int64
  integer_size_necessary::Int64
  real_size_factors::Int64
  integer_size_factors::Int64
  entries_in_factors::Int64
  max_task_pool_size::INT
  max_front_size::INT
  compresses_real::INT
  compresses_integer::INT
  two_by_two_pivots::INT
  semi_bandwidth::INT
  delayed_pivots::INT
  pivot_sign_changes::INT
  static_pivots::INT
  first_modified_pivot::INT
  rank::INT
  negative_eigenvalues::INT
  num_zero::INT
  iterative_refinements::INT
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
  sils_ainfo::sils_ainfo_type{T,INT}
  sils_finfo::sils_finfo_type{T,INT}
  sils_sinfo::sils_sinfo_type{T,INT}
  ma57_ainfo::ma57_ainfo{T,INT}
  ma57_finfo::ma57_finfo{T,INT}
  ma57_sinfo::ma57_sinfo{T,INT}
  ma77_info::ma77_info{T,INT}
  ma86_info::ma86_info{T,INT}
  ma87_info::ma87_info{T,INT}
  ma97_info::ma97_info{T,INT}
  ssids_inform::spral_ssids_inform{INT}
  mc61_info::NTuple{10,INT}
  mc61_rinfo::NTuple{15,T}
  mc64_info::mc64_info{INT}
  mc68_info::mc68_info{INT}
  mc77_info::NTuple{10,INT}
  mc77_rinfo::NTuple{10,T}
  mumps_error::INT
  mumps_info::NTuple{80,INT}
  mumps_rinfo::NTuple{40,T}
  pardiso_error::INT
  pardiso_IPARM::NTuple{64,INT}
  pardiso_DPARM::NTuple{64,T}
  mkl_pardiso_error::INT
  mkl_pardiso_IPARM::NTuple{64,INT}
  pastix_info::INT
  wsmp_error::INT
  wsmp_iparm::NTuple{64,INT}
  wsmp_dparm::NTuple{64,T}
  mpi_ierr::INT
  lapack_error::INT
end

export sls_initialize

function sls_initialize(::Type{Float32}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_single.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sls_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sls_initialize(::Type{Float32}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_single_64.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sls_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sls_initialize(::Type{Float64}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_double.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sls_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sls_initialize(::Type{Float64}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_double_64.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sls_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sls_initialize(::Type{Float128}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_quadruple.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sls_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function sls_initialize(::Type{Float128}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_quadruple_64.sls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sls_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export sls_read_specfile

function sls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.sls_read_specfile(control::Ptr{sls_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function sls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.sls_read_specfile(control::Ptr{sls_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.sls_read_specfile(control::Ptr{sls_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function sls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.sls_read_specfile(control::Ptr{sls_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.sls_read_specfile(control::Ptr{sls_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.sls_read_specfile(control::Ptr{sls_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export sls_analyse_matrix

function sls_analyse_matrix(::Type{Float32}, ::Type{Int32}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_single.sls_analyse_matrix(control::Ptr{sls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, type::Ptr{Cchar}, ne::Int32,
                                              row::Ptr{Int32}, col::Ptr{Int32},
                                              ptr::Ptr{Int32})::Cvoid
end

function sls_analyse_matrix(::Type{Float32}, ::Type{Int64}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_single_64.sls_analyse_matrix(control::Ptr{sls_control_type{Float32,
                                                                               Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, type::Ptr{Cchar}, ne::Int64,
                                                 row::Ptr{Int64}, col::Ptr{Int64},
                                                 ptr::Ptr{Int64})::Cvoid
end

function sls_analyse_matrix(::Type{Float64}, ::Type{Int32}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_double.sls_analyse_matrix(control::Ptr{sls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, type::Ptr{Cchar}, ne::Int32,
                                              row::Ptr{Int32}, col::Ptr{Int32},
                                              ptr::Ptr{Int32})::Cvoid
end

function sls_analyse_matrix(::Type{Float64}, ::Type{Int64}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_double_64.sls_analyse_matrix(control::Ptr{sls_control_type{Float64,
                                                                               Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, type::Ptr{Cchar}, ne::Int64,
                                                 row::Ptr{Int64}, col::Ptr{Int64},
                                                 ptr::Ptr{Int64})::Cvoid
end

function sls_analyse_matrix(::Type{Float128}, ::Type{Int32}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_quadruple.sls_analyse_matrix(control::Ptr{sls_control_type{Float128,
                                                                               Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, type::Ptr{Cchar}, ne::Int32,
                                                 row::Ptr{Int32}, col::Ptr{Int32},
                                                 ptr::Ptr{Int32})::Cvoid
end

function sls_analyse_matrix(::Type{Float128}, ::Type{Int64}, control, data, status, n, type,
                            ne, row, col, ptr)
  @ccall libgalahad_quadruple_64.sls_analyse_matrix(control::Ptr{sls_control_type{Float128,
                                                                                  Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    type::Ptr{Cchar}, ne::Int64,
                                                    row::Ptr{Int64}, col::Ptr{Int64},
                                                    ptr::Ptr{Int64})::Cvoid
end

export sls_reset_control

function sls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.sls_reset_control(control::Ptr{sls_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function sls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.sls_reset_control(control::Ptr{sls_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function sls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.sls_reset_control(control::Ptr{sls_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function sls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.sls_reset_control(control::Ptr{sls_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function sls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.sls_reset_control(control::Ptr{sls_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function sls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.sls_reset_control(control::Ptr{sls_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export sls_factorize_matrix

function sls_factorize_matrix(::Type{Float32}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_single.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                ne::Int32, val::Ptr{Float32})::Cvoid
end

function sls_factorize_matrix(::Type{Float32}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_single_64.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, ne::Int64,
                                                   val::Ptr{Float32})::Cvoid
end

function sls_factorize_matrix(::Type{Float64}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_double.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                ne::Int32, val::Ptr{Float64})::Cvoid
end

function sls_factorize_matrix(::Type{Float64}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_double_64.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, ne::Int64,
                                                   val::Ptr{Float64})::Cvoid
end

function sls_factorize_matrix(::Type{Float128}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_quadruple.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, ne::Int32,
                                                   val::Ptr{Float128})::Cvoid
end

function sls_factorize_matrix(::Type{Float128}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_quadruple_64.sls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, ne::Int64,
                                                      val::Ptr{Float128})::Cvoid
end

export sls_solve_system

function sls_solve_system(::Type{Float32}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_single.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            n::Int32, sol::Ptr{Float32})::Cvoid
end

function sls_solve_system(::Type{Float32}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_single_64.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               n::Int64, sol::Ptr{Float32})::Cvoid
end

function sls_solve_system(::Type{Float64}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_double.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            n::Int32, sol::Ptr{Float64})::Cvoid
end

function sls_solve_system(::Type{Float64}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_double_64.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               n::Int64, sol::Ptr{Float64})::Cvoid
end

function sls_solve_system(::Type{Float128}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_quadruple.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                               n::Int32, sol::Ptr{Float128})::Cvoid
end

function sls_solve_system(::Type{Float128}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_quadruple_64.sls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                  n::Int64, sol::Ptr{Float128})::Cvoid
end

export sls_partial_solve_system

function sls_partial_solve_system(::Type{Float32}, ::Type{Int32}, part, data, status, n,
                                  sol)
  @ccall libgalahad_single.sls_partial_solve_system(part::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    sol::Ptr{Float32})::Cvoid
end

function sls_partial_solve_system(::Type{Float32}, ::Type{Int64}, part, data, status, n,
                                  sol)
  @ccall libgalahad_single_64.sls_partial_solve_system(part::Ptr{Cchar},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       sol::Ptr{Float32})::Cvoid
end

function sls_partial_solve_system(::Type{Float64}, ::Type{Int32}, part, data, status, n,
                                  sol)
  @ccall libgalahad_double.sls_partial_solve_system(part::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    sol::Ptr{Float64})::Cvoid
end

function sls_partial_solve_system(::Type{Float64}, ::Type{Int64}, part, data, status, n,
                                  sol)
  @ccall libgalahad_double_64.sls_partial_solve_system(part::Ptr{Cchar},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       sol::Ptr{Float64})::Cvoid
end

function sls_partial_solve_system(::Type{Float128}, ::Type{Int32}, part, data, status, n,
                                  sol)
  @ccall libgalahad_quadruple.sls_partial_solve_system(part::Ptr{Cchar},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32}, n::Int32,
                                                       sol::Ptr{Float128})::Cvoid
end

function sls_partial_solve_system(::Type{Float128}, ::Type{Int64}, part, data, status, n,
                                  sol)
  @ccall libgalahad_quadruple_64.sls_partial_solve_system(part::Ptr{Cchar},
                                                          data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64}, n::Int64,
                                                          sol::Ptr{Float128})::Cvoid
end

export sls_information

function sls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.sls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sls_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.sls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sls_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.sls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sls_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.sls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sls_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.sls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sls_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.sls_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{sls_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export sls_terminate

function sls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sls_control_type{Float32,Int32}},
                                         inform::Ptr{sls_inform_type{Float32,Int32}})::Cvoid
end

function sls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sls_control_type{Float32,Int64}},
                                            inform::Ptr{sls_inform_type{Float32,Int64}})::Cvoid
end

function sls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sls_control_type{Float64,Int32}},
                                         inform::Ptr{sls_inform_type{Float64,Int32}})::Cvoid
end

function sls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sls_control_type{Float64,Int64}},
                                            inform::Ptr{sls_inform_type{Float64,Int64}})::Cvoid
end

function sls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sls_control_type{Float128,Int32}},
                                            inform::Ptr{sls_inform_type{Float128,Int32}})::Cvoid
end

function sls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.sls_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{sls_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{sls_inform_type{Float128,Int64}})::Cvoid
end
