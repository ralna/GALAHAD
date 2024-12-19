export qpa_control_type

struct qpa_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  maxit::Cint
  factor::Cint
  max_col::Cint
  max_sc::Cint
  indmin::Cint
  valmin::Cint
  itref_max::Cint
  infeas_check_interval::Cint
  cg_maxit::Cint
  precon::Cint
  nsemib::Cint
  full_max_fill::Cint
  deletion_strategy::Cint
  restore_problem::Cint
  monitor_residuals::Cint
  cold_start::Cint
  sif_file_device::Cint
  infinity::T
  feas_tol::T
  obj_unbounded::T
  increase_rho_g_factor::T
  infeas_g_improved_by_factor::T
  increase_rho_b_factor::T
  infeas_b_improved_by_factor::T
  pivot_tol::T
  pivot_tol_for_dependencies::T
  zero_pivot::T
  inner_stop_relative::T
  inner_stop_absolute::T
  multiplier_tol::T
  cpu_time_limit::T
  clock_time_limit::T
  treat_zero_bounds_as_general::Bool
  solve_qp::Bool
  solve_within_bounds::Bool
  randomize::Bool
  array_syntax_worse_than_do_loop::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  each_interval::Bool
  sls_control::sls_control_type{T}
end

export qpa_time_type

struct qpa_time_type{T}
  total::T
  preprocess::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_preprocess::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export qpa_inform_type

struct qpa_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  major_iter::Cint
  iter::Cint
  cg_iter::Cint
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::Cint
  nmods::Cint
  num_g_infeas::Cint
  num_b_infeas::Cint
  obj::T
  infeas_g::T
  infeas_b::T
  merit::T
  time::qpa_time_type{T}
  sls_inform::sls_inform_type{T}
end

export qpa_initialize

function qpa_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.qpa_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{qpa_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function qpa_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.qpa_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{qpa_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function qpa_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.qpa_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{qpa_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export qpa_read_specfile

function qpa_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.qpa_read_specfile_s(control::Ptr{qpa_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function qpa_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.qpa_read_specfile(control::Ptr{qpa_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function qpa_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.qpa_read_specfile_q(control::Ptr{qpa_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export qpa_import

function qpa_import(::Type{Float32}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.qpa_import_s(control::Ptr{qpa_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function qpa_import(::Type{Float64}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.qpa_import(control::Ptr{qpa_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                      H_row::Ptr{Cint}, H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                      A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                      A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function qpa_import(::Type{Float128}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.qpa_import_q(control::Ptr{qpa_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                           H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                           H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                           A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                           A_ptr::Ptr{Cint})::Cvoid
end

export qpa_reset_control

function qpa_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.qpa_reset_control_s(control::Ptr{qpa_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function qpa_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.qpa_reset_control(control::Ptr{qpa_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function qpa_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.qpa_reset_control_q(control::Ptr{qpa_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export qpa_solve_qp

function qpa_solve_qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.qpa_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                          g::Ptr{Float32}, f::Float32, a_ne::Cint,
                                          A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, x::Ptr{Float32},
                                          c::Ptr{Float32}, y::Ptr{Float32}, z::Ptr{Float32},
                                          x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.qpa_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y::Ptr{Float64}, z::Ptr{Float64}, x_stat::Ptr{Cint},
                                        c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.qpa_solve_qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, h_ne::Cint,
                                             H_val::Ptr{Float128}, g::Ptr{Float128},
                                             f::Cfloat128, a_ne::Cint, A_val::Ptr{Float128},
                                             c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                             x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                             x::Ptr{Float128}, c::Ptr{Float128},
                                             y::Ptr{Float128}, z::Ptr{Float128},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export qpa_solve_l1qp

function qpa_solve_l1qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                        rho_b, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.qpa_solve_l1qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, h_ne::Cint,
                                            H_val::Ptr{Float32}, g::Ptr{Float32},
                                            f::Float32, rho_g::Float32, rho_b::Float32,
                                            a_ne::Cint, A_val::Ptr{Float32},
                                            c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                            x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                            x::Ptr{Float32}, c::Ptr{Float32},
                                            y::Ptr{Float32}, z::Ptr{Float32},
                                            x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_l1qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                        rho_b, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.qpa_solve_l1qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, rho_g::Float64,
                                          rho_b::Float64, a_ne::Cint, A_val::Ptr{Float64},
                                          c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                          x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                          x::Ptr{Float64}, c::Ptr{Float64}, y::Ptr{Float64},
                                          z::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_l1qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                        rho_b, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.qpa_solve_l1qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, m::Cint, h_ne::Cint,
                                               H_val::Ptr{Float128}, g::Ptr{Float128},
                                               f::Cfloat128, rho_g::Cfloat128,
                                               rho_b::Cfloat128, a_ne::Cint,
                                               A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                               c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                               x_u::Ptr{Float128}, x::Ptr{Float128},
                                               c::Ptr{Float128}, y::Ptr{Float128},
                                               z::Ptr{Float128}, x_stat::Ptr{Cint},
                                               c_stat::Ptr{Cint})::Cvoid
end

export qpa_solve_bcl1qp

function qpa_solve_bcl1qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                          a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.qpa_solve_bcl1qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, m::Cint, h_ne::Cint,
                                              H_val::Ptr{Float32}, g::Ptr{Float32},
                                              f::Float32, rho_g::Float32, a_ne::Cint,
                                              A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                              c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                              x_u::Ptr{Float32}, x::Ptr{Float32},
                                              c::Ptr{Float32}, y::Ptr{Float32},
                                              z::Ptr{Float32}, x_stat::Ptr{Cint},
                                              c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_bcl1qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                          a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.qpa_solve_bcl1qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, h_ne::Cint,
                                            H_val::Ptr{Float64}, g::Ptr{Float64},
                                            f::Float64, rho_g::Float64, a_ne::Cint,
                                            A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                            c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                            x_u::Ptr{Float64}, x::Ptr{Float64},
                                            c::Ptr{Float64}, y::Ptr{Float64},
                                            z::Ptr{Float64}, x_stat::Ptr{Cint},
                                            c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_bcl1qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, rho_g,
                          a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.qpa_solve_bcl1qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, m::Cint, h_ne::Cint,
                                                 H_val::Ptr{Float128}, g::Ptr{Float128},
                                                 f::Cfloat128, rho_g::Cfloat128, a_ne::Cint,
                                                 A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                                 c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                                 x_u::Ptr{Float128}, x::Ptr{Float128},
                                                 c::Ptr{Float128}, y::Ptr{Float128},
                                                 z::Ptr{Float128}, x_stat::Ptr{Cint},
                                                 c_stat::Ptr{Cint})::Cvoid
end

export qpa_information

function qpa_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.qpa_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{qpa_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function qpa_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.qpa_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{qpa_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function qpa_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.qpa_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{qpa_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export qpa_terminate

function qpa_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.qpa_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{qpa_control_type{Float32}},
                                           inform::Ptr{qpa_inform_type{Float32}})::Cvoid
end

function qpa_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.qpa_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{qpa_control_type{Float64}},
                                         inform::Ptr{qpa_inform_type{Float64}})::Cvoid
end

function qpa_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.qpa_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{qpa_control_type{Float128}},
                                              inform::Ptr{qpa_inform_type{Float128}})::Cvoid
end
