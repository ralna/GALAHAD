export dqp_control_type

struct dqp_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  dual_starting_point::INT
  maxit::INT
  max_sc::INT
  cauchy_only::INT
  arc_search_maxit::INT
  cg_maxit::INT
  explore_optimal_subspace::INT
  restore_problem::INT
  sif_file_device::INT
  qplib_file_device::INT
  rho::T
  infinity::T
  stop_abs_p::T
  stop_rel_p::T
  stop_abs_d::T
  stop_rel_d::T
  stop_abs_c::T
  stop_rel_c::T
  stop_cg_relative::T
  stop_cg_absolute::T
  cg_zero_curvature::T
  max_growth::T
  identical_bounds_tol::T
  cpu_time_limit::T
  clock_time_limit::T
  initial_perturbation::T
  perturbation_reduction::T
  final_perturbation::T
  factor_optimal_matrix::Bool
  remove_dependencies::Bool
  treat_zero_bounds_as_general::Bool
  exact_arc_search::Bool
  subspace_direct::Bool
  subspace_alternate::Bool
  subspace_arc_search::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  generate_qplib_file::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  definite_linear_solver::NTuple{31,Cchar}
  unsymmetric_linear_solver::NTuple{31,Cchar}
  sif_file_name::NTuple{31,Cchar}
  qplib_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  fdc_control::fdc_control_type{T,INT}
  sls_control::sls_control_type{T,INT}
  sbls_control::sbls_control_type{T,INT}
  gltr_control::gltr_control_type{T,INT}
end

export dqp_time_type

struct dqp_time_type{T}
  total::T
  preprocess::T
  find_dependent::T
  analyse::T
  factorize::T
  solve::T
  search::T
  clock_total::T
  clock_preprocess::T
  clock_find_dependent::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
  clock_search::T
end

export dqp_inform_type

struct dqp_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  cg_iter::INT
  factorization_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::INT
  threads::INT
  obj::T
  primal_infeasibility::T
  dual_infeasibility::T
  complementary_slackness::T
  non_negligible_pivot::T
  feasible::Bool
  checkpointsIter::NTuple{16,INT}
  checkpointsTime::NTuple{16,T}
  time::dqp_time_type{T}
  fdc_inform::fdc_inform_type{T,INT}
  sls_inform::sls_inform_type{T,INT}
  sbls_inform::sbls_inform_type{T,INT}
  gltr_inform::gltr_inform_type{T,INT}
  scu_status::INT
  scu_inform::scu_inform_type{INT}
  rpd_inform::rpd_inform_type{INT}
end

export dqp_initialize

function dqp_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dqp_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dqp_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dqp_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dqp_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dqp_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dqp_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dqp_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dqp_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dqp_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function dqp_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{dqp_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export dqp_read_specfile

function dqp_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.dqp_read_specfile(control::Ptr{dqp_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dqp_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.dqp_read_specfile(control::Ptr{dqp_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dqp_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.dqp_read_specfile(control::Ptr{dqp_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dqp_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.dqp_read_specfile(control::Ptr{dqp_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dqp_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.dqp_read_specfile(control::Ptr{dqp_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dqp_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.dqp_read_specfile(control::Ptr{dqp_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export dqp_import

function dqp_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.dqp_import(control::Ptr{dqp_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                      H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                      H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function dqp_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single_64.dqp_import(control::Ptr{dqp_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, H_type::Ptr{Cchar},
                                         H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64}, A_type::Ptr{Cchar}, A_ne::Int64,
                                         A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function dqp_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.dqp_import(control::Ptr{dqp_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                      H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                      H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function dqp_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double_64.dqp_import(control::Ptr{dqp_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, H_type::Ptr{Cchar},
                                         H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64}, A_type::Ptr{Cchar}, A_ne::Int64,
                                         A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function dqp_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.dqp_import(control::Ptr{dqp_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, m::Int32, H_type::Ptr{Cchar},
                                         H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                         A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                         A_ptr::Ptr{Int32})::Cvoid
end

function dqp_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple_64.dqp_import(control::Ptr{dqp_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, m::Int64, H_type::Ptr{Cchar},
                                            H_ne::Int64, H_row::Ptr{Int64},
                                            H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                            A_type::Ptr{Cchar}, A_ne::Int64,
                                            A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                            A_ptr::Ptr{Int64})::Cvoid
end

export dqp_reset_control

function dqp_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.dqp_reset_control(control::Ptr{dqp_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dqp_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.dqp_reset_control(control::Ptr{dqp_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dqp_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.dqp_reset_control(control::Ptr{dqp_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dqp_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.dqp_reset_control(control::Ptr{dqp_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dqp_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.dqp_reset_control(control::Ptr{dqp_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function dqp_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.dqp_reset_control(control::Ptr{dqp_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export dqp_solve_qp

function dqp_solve_qp(::Type{Float32}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, h_ne::Int32, H_val::Ptr{Float32},
                                        g::Ptr{Float32}, f::Float32, a_ne::Int32,
                                        A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                        c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                        x_u::Ptr{Float32}, x::Ptr{Float32}, c::Ptr{Float32},
                                        y::Ptr{Float32}, z::Ptr{Float32},
                                        x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_qp(::Type{Float32}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single_64.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, h_ne::Int64,
                                           H_val::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                           a_ne::Int64, A_val::Ptr{Float32},
                                           c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                           x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                           x::Ptr{Float32}, c::Ptr{Float32},
                                           y::Ptr{Float32}, z::Ptr{Float32},
                                           x_stat::Ptr{Int64}, c_stat::Ptr{Int64})::Cvoid
end

function dqp_solve_qp(::Type{Float64}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, h_ne::Int32, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Int32,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y::Ptr{Float64}, z::Ptr{Float64},
                                        x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_qp(::Type{Float64}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double_64.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, h_ne::Int64,
                                           H_val::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Int64, A_val::Ptr{Float64},
                                           c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                           x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                           x::Ptr{Float64}, c::Ptr{Float64},
                                           y::Ptr{Float64}, z::Ptr{Float64},
                                           x_stat::Ptr{Int64}, c_stat::Ptr{Int64})::Cvoid
end

function dqp_solve_qp(::Type{Float128}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g,
                      f, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, h_ne::Int32,
                                           H_val::Ptr{Float128}, g::Ptr{Float128},
                                           f::Cfloat128, a_ne::Int32, A_val::Ptr{Float128},
                                           c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                           x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                           x::Ptr{Float128}, c::Ptr{Float128},
                                           y::Ptr{Float128}, z::Ptr{Float128},
                                           x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_qp(::Type{Float128}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g,
                      f, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple_64.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, h_ne::Int64,
                                              H_val::Ptr{Float128}, g::Ptr{Float128},
                                              f::Cfloat128, a_ne::Int64,
                                              A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                              c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                              x_u::Ptr{Float128}, x::Ptr{Float128},
                                              c::Ptr{Float128}, y::Ptr{Float128},
                                              z::Ptr{Float128}, x_stat::Ptr{Int64},
                                              c_stat::Ptr{Int64})::Cvoid
end

export dqp_solve_sldqp

function dqp_solve_sldqp(::Type{Float32}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, w::Ptr{Float32},
                                           x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                           a_ne::Int32, A_val::Ptr{Float32},
                                           c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                           x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                           x::Ptr{Float32}, c::Ptr{Float32},
                                           y::Ptr{Float32}, z::Ptr{Float32},
                                           x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_sldqp(::Type{Float32}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single_64.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, w::Ptr{Float32},
                                              x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                              a_ne::Int64, A_val::Ptr{Float32},
                                              c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                              x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                              x::Ptr{Float32}, c::Ptr{Float32},
                                              y::Ptr{Float32}, z::Ptr{Float32},
                                              x_stat::Ptr{Int64}, c_stat::Ptr{Int64})::Cvoid
end

function dqp_solve_sldqp(::Type{Float64}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, w::Ptr{Float64},
                                           x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Int32, A_val::Ptr{Float64},
                                           c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                           x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                           x::Ptr{Float64}, c::Ptr{Float64},
                                           y::Ptr{Float64}, z::Ptr{Float64},
                                           x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_sldqp(::Type{Float64}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double_64.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, w::Ptr{Float64},
                                              x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                              a_ne::Int64, A_val::Ptr{Float64},
                                              c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                              x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                              x::Ptr{Float64}, c::Ptr{Float64},
                                              y::Ptr{Float64}, z::Ptr{Float64},
                                              x_stat::Ptr{Int64}, c_stat::Ptr{Int64})::Cvoid
end

function dqp_solve_sldqp(::Type{Float128}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, m::Int32, w::Ptr{Float128},
                                              x0::Ptr{Float128}, g::Ptr{Float128},
                                              f::Cfloat128, a_ne::Int32,
                                              A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                              c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                              x_u::Ptr{Float128}, x::Ptr{Float128},
                                              c::Ptr{Float128}, y::Ptr{Float128},
                                              z::Ptr{Float128}, x_stat::Ptr{Int32},
                                              c_stat::Ptr{Int32})::Cvoid
end

function dqp_solve_sldqp(::Type{Float128}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple_64.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, m::Int64, w::Ptr{Float128},
                                                 x0::Ptr{Float128}, g::Ptr{Float128},
                                                 f::Cfloat128, a_ne::Int64,
                                                 A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                                 c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                                 x_u::Ptr{Float128}, x::Ptr{Float128},
                                                 c::Ptr{Float128}, y::Ptr{Float128},
                                                 z::Ptr{Float128}, x_stat::Ptr{Int64},
                                                 c_stat::Ptr{Int64})::Cvoid
end

export dqp_information

function dqp_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.dqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dqp_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dqp_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.dqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dqp_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dqp_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.dqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dqp_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dqp_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.dqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dqp_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dqp_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.dqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dqp_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function dqp_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.dqp_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{dqp_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export dqp_terminate

function dqp_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dqp_control_type{Float32,Int32}},
                                         inform::Ptr{dqp_inform_type{Float32,Int32}})::Cvoid
end

function dqp_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dqp_control_type{Float32,Int64}},
                                            inform::Ptr{dqp_inform_type{Float32,Int64}})::Cvoid
end

function dqp_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dqp_control_type{Float64,Int32}},
                                         inform::Ptr{dqp_inform_type{Float64,Int32}})::Cvoid
end

function dqp_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dqp_control_type{Float64,Int64}},
                                            inform::Ptr{dqp_inform_type{Float64,Int64}})::Cvoid
end

function dqp_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dqp_control_type{Float128,Int32}},
                                            inform::Ptr{dqp_inform_type{Float128,Int32}})::Cvoid
end

function dqp_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{dqp_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{dqp_inform_type{Float128,Int64}})::Cvoid
end
