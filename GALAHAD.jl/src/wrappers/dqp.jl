export dqp_control_type

mutable struct dqp_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  dual_starting_point::Cint
  maxit::Cint
  max_sc::Cint
  cauchy_only::Cint
  arc_search_maxit::Cint
  cg_maxit::Cint
  explore_optimal_subspace::Cint
  restore_problem::Cint
  sif_file_device::Cint
  qplib_file_device::Cint
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
  fdc_control::fdc_control_type{T}
  sls_control::sls_control_type{T}
  sbls_control::sbls_control_type{T}
  gltr_control::gltr_control_type{T}

  function dqp_control_type{T}() where T
    type = new()
    type.fdc_control = fdc_control_type{T}()
    type.sls_control = sls_control_type{T}()
    type.sbls_control = sbls_control_type{T}()
    type.gltr_control = gltr_control_type{T}()
    return type
  end
end

export dqp_time_type

mutable struct dqp_time_type{T}
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

  dqp_time_type{T}() where T = new()
end

export dqp_inform_type

mutable struct dqp_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  cg_iter::Cint
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::Cint
  threads::Cint
  obj::T
  primal_infeasibility::T
  dual_infeasibility::T
  complementary_slackness::T
  non_negligible_pivot::T
  feasible::Bool
  checkpointsIter::NTuple{16,Cint}
  checkpointsTime::NTuple{16,T}
  time::dqp_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sls_inform::sls_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  gltr_inform::gltr_inform_type{T}
  scu_status::Cint
  scu_inform::scu_inform_type
  rpd_inform::rpd_inform_type

  function dqp_inform_type{T}() where T
    type = new()
    type.time = dqp_time_type{T}()
    type.fdc_inform = fdc_inform_type{T}()
    type.sls_inform = sls_inform_type{T}()
    type.sbls_inform = sbls_inform_type{T}()
    type.gltr_inform = gltr_inform_type{T}()
    type.scu_inform = scu_inform_type()
    type.rpd_inform = rpd_inform_type()
    return type
  end
end

export dqp_initialize_s

function dqp_initialize_s(data, control, status)
  @ccall libgalahad_single.dqp_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{dqp_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export dqp_initialize

function dqp_initialize(data, control, status)
  @ccall libgalahad_double.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{dqp_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export dqp_read_specfile_s

function dqp_read_specfile_s(control, specfile)
  @ccall libgalahad_single.dqp_read_specfile_s(control::Ref{dqp_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export dqp_read_specfile

function dqp_read_specfile(control, specfile)
  @ccall libgalahad_double.dqp_read_specfile(control::Ref{dqp_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export dqp_import_s

function dqp_import_s(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.dqp_import_s(control::Ref{dqp_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

export dqp_import

function dqp_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                  A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.dqp_import(control::Ref{dqp_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                      H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                      H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                      A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                      A_ptr::Ptr{Cint})::Cvoid
end

export dqp_reset_control_s

function dqp_reset_control_s(control, data, status)
  @ccall libgalahad_single.dqp_reset_control_s(control::Ref{dqp_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export dqp_reset_control

function dqp_reset_control(control, data, status)
  @ccall libgalahad_double.dqp_reset_control(control::Ref{dqp_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export dqp_solve_qp_s

function dqp_solve_qp_s(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.dqp_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                          g::Ptr{Float32}, f::Float32, a_ne::Cint,
                                          A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, x::Ptr{Float32},
                                          c::Ptr{Float32}, y::Ptr{Float32},
                                          z::Ptr{Float32}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

export dqp_solve_qp

function dqp_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                    x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64},
                                        c::Ptr{Float64}, y::Ptr{Float64},
                                        z::Ptr{Float64}, x_stat::Ptr{Cint},
                                        c_stat::Ptr{Cint})::Cvoid
end

export dqp_solve_sldqp_s

function dqp_solve_sldqp_s(data, status, n, m, w, x0, g, f, a_ne, A_val, c_l, c_u, x_l, x_u,
                         x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.dqp_solve_sldqp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, w::Ptr{Float32},
                                             x0::Ptr{Float32}, g::Ptr{Float32},
                                             f::Float32, a_ne::Cint, A_val::Ptr{Float32},
                                             c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                             x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                             x::Ptr{Float32}, c::Ptr{Float32},
                                             y::Ptr{Float32}, z::Ptr{Float32},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export dqp_solve_sldqp

function dqp_solve_sldqp(data, status, n, m, w, x0, g, f, a_ne, A_val, c_l, c_u, x_l, x_u,
                       x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, w::Ptr{Float64},
                                           x0::Ptr{Float64}, g::Ptr{Float64},
                                           f::Float64, a_ne::Cint, A_val::Ptr{Float64},
                                           c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                           x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                           x::Ptr{Float64}, c::Ptr{Float64},
                                           y::Ptr{Float64}, z::Ptr{Float64},
                                           x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export dqp_information_s

function dqp_information_s(data, inform, status)
  @ccall libgalahad_single.dqp_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{dqp_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export dqp_information

function dqp_information(data, inform, status)
  @ccall libgalahad_double.dqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{dqp_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export dqp_terminate_s

function dqp_terminate_s(data, control, inform)
  @ccall libgalahad_single.dqp_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{dqp_control_type{Float32}},
                                           inform::Ref{dqp_inform_type{Float32}})::Cvoid
end

export dqp_terminate

function dqp_terminate(data, control, inform)
  @ccall libgalahad_double.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{dqp_control_type{Float64}},
                                         inform::Ref{dqp_inform_type{Float64}})::Cvoid
end
