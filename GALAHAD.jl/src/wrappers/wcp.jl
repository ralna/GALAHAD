export wcp_control_type

struct wcp_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  maxit::INT
  initial_point::INT
  factor::INT
  max_col::INT
  indmin::INT
  valmin::INT
  itref_max::INT
  infeas_max::INT
  perturbation_strategy::INT
  restore_problem::INT
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
  fdc_control::fdc_control_type{T,INT}
  sbls_control::sbls_control_type{T,INT}
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

struct wcp_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  factorization_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::INT
  c_implicit::INT
  x_implicit::INT
  y_implicit::INT
  z_implicit::INT
  obj::T
  mu_final_target_max::T
  non_negligible_pivot::T
  feasible::Bool
  time::wcp_time_type{T}
  fdc_inform::fdc_inform_type{T,INT}
  sbls_inform::sbls_inform_type{T,INT}
end

export wcp_initialize

function wcp_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{wcp_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function wcp_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{wcp_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function wcp_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{wcp_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function wcp_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{wcp_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function wcp_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{wcp_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function wcp_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{wcp_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export wcp_read_specfile

function wcp_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.wcp_read_specfile(control::Ptr{wcp_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.wcp_read_specfile(control::Ptr{wcp_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.wcp_read_specfile(control::Ptr{wcp_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.wcp_read_specfile(control::Ptr{wcp_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.wcp_read_specfile(control::Ptr{wcp_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function wcp_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.wcp_read_specfile(control::Ptr{wcp_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export wcp_import

function wcp_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.wcp_import(control::Ptr{wcp_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function wcp_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single_64.wcp_import(control::Ptr{wcp_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, A_type::Ptr{Cchar},
                                         A_ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function wcp_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.wcp_import(control::Ptr{wcp_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function wcp_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double_64.wcp_import(control::Ptr{wcp_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, A_type::Ptr{Cchar},
                                         A_ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function wcp_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.wcp_import(control::Ptr{wcp_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, m::Int32, A_type::Ptr{Cchar},
                                         A_ne::Int32, A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                         A_ptr::Ptr{Int32})::Cvoid
end

function wcp_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, A_type,
                    A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple_64.wcp_import(control::Ptr{wcp_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, m::Int64, A_type::Ptr{Cchar},
                                            A_ne::Int64, A_row::Ptr{Int64},
                                            A_col::Ptr{Int64}, A_ptr::Ptr{Int64})::Cvoid
end

export wcp_reset_control

function wcp_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.wcp_reset_control(control::Ptr{wcp_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function wcp_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.wcp_reset_control(control::Ptr{wcp_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function wcp_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.wcp_reset_control(control::Ptr{wcp_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function wcp_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.wcp_reset_control(control::Ptr{wcp_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function wcp_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.wcp_reset_control(control::Ptr{wcp_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function wcp_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.wcp_reset_control(control::Ptr{wcp_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export wcp_find_wcp

function wcp_find_wcp(::Type{Float32}, ::Type{Int32}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_single.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, g::Ptr{Float32}, a_ne::Int32,
                                        A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                        c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                        x_u::Ptr{Float32}, x::Ptr{Float32}, c::Ptr{Float32},
                                        y_l::Ptr{Float32}, y_u::Ptr{Float32},
                                        z_l::Ptr{Float32}, z_u::Ptr{Float32},
                                        x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function wcp_find_wcp(::Type{Float32}, ::Type{Int64}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_single_64.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, g::Ptr{Float32}, a_ne::Int64,
                                           A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                           c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                           x_u::Ptr{Float32}, x::Ptr{Float32},
                                           c::Ptr{Float32}, y_l::Ptr{Float32},
                                           y_u::Ptr{Float32}, z_l::Ptr{Float32},
                                           z_u::Ptr{Float32}, x_stat::Ptr{Int64},
                                           c_stat::Ptr{Int64})::Cvoid
end

function wcp_find_wcp(::Type{Float64}, ::Type{Int32}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_double.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, g::Ptr{Float64}, a_ne::Int32,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y_l::Ptr{Float64}, y_u::Ptr{Float64},
                                        z_l::Ptr{Float64}, z_u::Ptr{Float64},
                                        x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function wcp_find_wcp(::Type{Float64}, ::Type{Int64}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_double_64.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, g::Ptr{Float64}, a_ne::Int64,
                                           A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                           c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                           x_u::Ptr{Float64}, x::Ptr{Float64},
                                           c::Ptr{Float64}, y_l::Ptr{Float64},
                                           y_u::Ptr{Float64}, z_l::Ptr{Float64},
                                           z_u::Ptr{Float64}, x_stat::Ptr{Int64},
                                           c_stat::Ptr{Int64})::Cvoid
end

function wcp_find_wcp(::Type{Float128}, ::Type{Int32}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_quadruple.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, g::Ptr{Float128},
                                           a_ne::Int32, A_val::Ptr{Float128},
                                           c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                           x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                           x::Ptr{Float128}, c::Ptr{Float128},
                                           y_l::Ptr{Float128}, y_u::Ptr{Float128},
                                           z_l::Ptr{Float128}, z_u::Ptr{Float128},
                                           x_stat::Ptr{Int32}, c_stat::Ptr{Int32})::Cvoid
end

function wcp_find_wcp(::Type{Float128}, ::Type{Int64}, data, status, n, m, g, a_ne, A_val,
                      c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u, x_stat, c_stat)
  @ccall libgalahad_quadruple_64.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, g::Ptr{Float128},
                                              a_ne::Int64, A_val::Ptr{Float128},
                                              c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                              x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                              x::Ptr{Float128}, c::Ptr{Float128},
                                              y_l::Ptr{Float128}, y_u::Ptr{Float128},
                                              z_l::Ptr{Float128}, z_u::Ptr{Float128},
                                              x_stat::Ptr{Int64}, c_stat::Ptr{Int64})::Cvoid
end

export wcp_information

function wcp_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.wcp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{wcp_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function wcp_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.wcp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{wcp_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function wcp_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.wcp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{wcp_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function wcp_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.wcp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{wcp_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function wcp_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.wcp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{wcp_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function wcp_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.wcp_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{wcp_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export wcp_terminate

function wcp_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{wcp_control_type{Float32,Int32}},
                                         inform::Ptr{wcp_inform_type{Float32,Int32}})::Cvoid
end

function wcp_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{wcp_control_type{Float32,Int64}},
                                            inform::Ptr{wcp_inform_type{Float32,Int64}})::Cvoid
end

function wcp_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{wcp_control_type{Float64,Int32}},
                                         inform::Ptr{wcp_inform_type{Float64,Int32}})::Cvoid
end

function wcp_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{wcp_control_type{Float64,Int64}},
                                            inform::Ptr{wcp_inform_type{Float64,Int64}})::Cvoid
end

function wcp_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{wcp_control_type{Float128,Int32}},
                                            inform::Ptr{wcp_inform_type{Float128,Int32}})::Cvoid
end

function wcp_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{wcp_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{wcp_inform_type{Float128,Int64}})::Cvoid
end
