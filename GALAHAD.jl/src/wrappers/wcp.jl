export wcp_control_type

struct wcp_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  maxit::Cint
  initial_point::Cint
  factor::Cint
  max_col::Cint
  indmin::Cint
  valmin::Cint
  itref_max::Cint
  infeas_max::Cint
  perturbation_strategy::Cint
  restore_problem::Cint
  infinity::T
  stop_p::T
  stop_d::T
  stop_c::T
  prfeas::T
  dufeas::T
  mu_target::T
  mu_accept_fraction::T
  mu_increase_factor::T
  required_infeas_reduction::T
  implicit_tol::T
  pivot_tol::T
  pivot_tol_for_dependencies::T
  zero_pivot::T
  perturb_start::T
  alpha_scale::T
  identical_bounds_tol::T
  reduce_perturb_factor::T
  reduce_perturb_multiplier::T
  insufficiently_feasible::T
  perturbation_small::T
  cpu_time_limit::T
  clock_time_limit::T
  remove_dependencies::Bool
  treat_zero_bounds_as_general::Bool
  just_feasible::Bool
  balance_initial_complementarity::Bool
  use_corrector::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  record_x_status::Bool
  record_c_status::Bool
  prefix::NTuple{31,Cchar}
  fdc_control::fdc_control_type{T}
  sbls_control::sbls_control_type{T}
end

export wcp_time_type

struct wcp_time_type{T}
  total::T
  preprocess::T
  find_dependent::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_preprocess::T
  clock_find_dependent::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export wcp_inform_type

struct wcp_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::Cint
  c_implicit::Cint
  x_implicit::Cint
  y_implicit::Cint
  z_implicit::Cint
  obj::T
  mu_final_target_max::T
  non_negligible_pivot::T
  feasible::Bool
  time::wcp_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sbls_inform::sbls_inform_type{T}
end

export wcp_initialize

function wcp_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.wcp_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{wcp_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function wcp_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{wcp_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export wcp_read_specfile

function wcp_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.wcp_read_specfile_s(control::Ptr{wcp_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.wcp_read_specfile(control::Ptr{wcp_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export wcp_import

function wcp_import(::Type{Float32}, control, data, status, n, m, A_type, A_ne, A_row,
                    A_col, A_ptr)
  @ccall libgalahad_single.wcp_import_s(control::Ptr{wcp_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function wcp_import(::Type{Float64}, control, data, status, n, m, A_type, A_ne, A_row,
                    A_col, A_ptr)
  @ccall libgalahad_double.wcp_import(control::Ptr{wcp_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                      A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                      A_ptr::Ptr{Cint})::Cvoid
end

export wcp_reset_control

function wcp_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.wcp_reset_control_s(control::Ptr{wcp_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function wcp_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.wcp_reset_control(control::Ptr{wcp_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export wcp_find_wcp

function wcp_find_wcp(::Type{Float32}, data, status, n, m, g, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_single.wcp_find_wcp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float32}, a_ne::Cint,
                                          A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, x::Ptr{Float32},
                                          c::Ptr{Float32}, y_l::Ptr{Float32},
                                          y_u::Ptr{Float32}, z_l::Ptr{Float32},
                                          z_u::Ptr{Float32}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function wcp_find_wcp(::Type{Float64}, data, status, n, m, g, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_double.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, g::Ptr{Float64}, a_ne::Cint,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y_l::Ptr{Float64}, y_u::Ptr{Float64},
                                        z_l::Ptr{Float64}, z_u::Ptr{Float64},
                                        x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export wcp_information

function wcp_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.wcp_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{wcp_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function wcp_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.wcp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{wcp_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export wcp_terminate

function wcp_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.wcp_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{wcp_control_type{Float32}},
                                           inform::Ptr{wcp_inform_type{Float32}})::Cvoid
end

function wcp_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{wcp_control_type{Float64}},
                                         inform::Ptr{wcp_inform_type{Float64}})::Cvoid
end
