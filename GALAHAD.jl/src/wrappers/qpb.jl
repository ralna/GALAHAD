export qpb_control_type

struct qpb_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  maxit::Cint
  itref_max::Cint
  cg_maxit::Cint
  indicator_type::Cint
  restore_problem::Cint
  extrapolate::Cint
  path_history::Cint
  factor::Cint
  max_col::Cint
  indmin::Cint
  valmin::Cint
  infeas_max::Cint
  precon::Cint
  nsemib::Cint
  path_derivatives::Cint
  fit_order::Cint
  sif_file_device::Cint
  infinity::T
  stop_p::T
  stop_d::T
  stop_c::T
  theta_d::T
  theta_c::T
  beta::T
  prfeas::T
  dufeas::T
  muzero::T
  reduce_infeas::T
  obj_unbounded::T
  pivot_tol::T
  pivot_tol_for_dependencies::T
  zero_pivot::T
  identical_bounds_tol::T
  inner_stop_relative::T
  inner_stop_absolute::T
  initial_radius::T
  mu_min::T
  inner_fraction_opt::T
  indicator_tol_p::T
  indicator_tol_pd::T
  indicator_tol_tapia::T
  cpu_time_limit::T
  clock_time_limit::T
  remove_dependencies::Bool
  treat_zero_bounds_as_general::Bool
  center::Bool
  primal::Bool
  puiseux::Bool
  feasol::Bool
  array_syntax_worse_than_do_loop::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  lsqp_control::lsqp_control_type{T}
  fdc_control::fdc_control_type{T}
  sbls_control::sbls_control_type{T}
  gltr_control::gltr_control_type{T}
  fit_control::fit_control_type
end

export qpb_time_type

struct qpb_time_type{T}
  total::T
  preprocess::T
  find_dependent::T
  analyse::T
  factorize::T
  solve::T
  phase1_total::T
  phase1_analyse::T
  phase1_factorize::T
  phase1_solve::T
  clock_total::T
  clock_preprocess::T
  clock_find_dependent::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
  clock_phase1_total::T
  clock_phase1_analyse::T
  clock_phase1_factorize::T
  clock_phase1_solve::T
end

export qpb_inform_type

struct qpb_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  cg_iter::Cint
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::Cint
  nbacts::Cint
  nmods::Cint
  obj::T
  non_negligible_pivot::T
  feasible::Bool
  time::qpb_time_type{T}
  lsqp_inform::lsqp_inform_type{T}
  fdc_inform::fdc_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  gltr_inform::gltr_inform_type{T}
  fit_inform::fit_inform_type
end

export qpb_initialize

function qpb_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.qpb_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{qpb_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function qpb_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.qpb_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{qpb_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function qpb_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.qpb_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{qpb_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export qpb_read_specfile

function qpb_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.qpb_read_specfile_s(control::Ptr{qpb_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function qpb_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.qpb_read_specfile(control::Ptr{qpb_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function qpb_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.qpb_read_specfile_q(control::Ptr{qpb_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export qpb_import

function qpb_import(::Type{Float32}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.qpb_import_s(control::Ptr{qpb_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function qpb_import(::Type{Float64}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.qpb_import(control::Ptr{qpb_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                      H_row::Ptr{Cint}, H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                      A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                      A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function qpb_import(::Type{Float128}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.qpb_import_q(control::Ptr{qpb_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                           H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                           H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                           A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                           A_ptr::Ptr{Cint})::Cvoid
end

export qpb_reset_control

function qpb_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.qpb_reset_control_s(control::Ptr{qpb_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function qpb_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.qpb_reset_control(control::Ptr{qpb_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function qpb_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.qpb_reset_control_q(control::Ptr{qpb_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export qpb_solve_qp

function qpb_solve_qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.qpb_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                          g::Ptr{Float32}, f::Float32, a_ne::Cint,
                                          A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, x::Ptr{Float32},
                                          c::Ptr{Float32}, y::Ptr{Float32}, z::Ptr{Float32},
                                          x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function qpb_solve_qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.qpb_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y::Ptr{Float64}, z::Ptr{Float64}, x_stat::Ptr{Cint},
                                        c_stat::Ptr{Cint})::Cvoid
end

function qpb_solve_qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.qpb_solve_qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, h_ne::Cint,
                                             H_val::Ptr{Float128}, g::Ptr{Float128},
                                             f::Cfloat128, a_ne::Cint, A_val::Ptr{Float128},
                                             c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                             x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                             x::Ptr{Float128}, c::Ptr{Float128},
                                             y::Ptr{Float128}, z::Ptr{Float128},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export qpb_information

function qpb_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.qpb_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{qpb_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function qpb_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.qpb_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{qpb_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function qpb_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.qpb_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{qpb_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export qpb_terminate

function qpb_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.qpb_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{qpb_control_type{Float32}},
                                           inform::Ptr{qpb_inform_type{Float32}})::Cvoid
end

function qpb_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.qpb_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{qpb_control_type{Float64}},
                                         inform::Ptr{qpb_inform_type{Float64}})::Cvoid
end

function qpb_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.qpb_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{qpb_control_type{Float128}},
                                              inform::Ptr{qpb_inform_type{Float128}})::Cvoid
end
