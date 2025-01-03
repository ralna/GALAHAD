export lpa_control_type

struct lpa_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  maxit::Cint
  max_iterative_refinements::Cint
  min_real_factor_size::Cint
  min_integer_factor_size::Cint
  random_number_seed::Cint
  sif_file_device::Cint
  qplib_file_device::Cint
  infinity::T
  tol_data::T
  feas_tol::T
  relative_pivot_tolerance::T
  growth_limit::T
  zero_tolerance::T
  change_tolerance::T
  identical_bounds_tol::T
  cpu_time_limit::T
  clock_time_limit::T
  scale::Bool
  dual::Bool
  warm_start::Bool
  steepest_edge::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  generate_qplib_file::Bool
  sif_file_name::NTuple{31,Cchar}
  qplib_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
end

export lpa_time_type

struct lpa_time_type{T}
  total::T
  preprocess::T
  clock_total::T
  clock_preprocess::T
end

export lpa_inform_type

struct lpa_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  la04_job::Cint
  la04_job_info::Cint
  obj::T
  primal_infeasibility::T
  feasible::Bool
  RINFO::NTuple{40,T}
  time::lpa_time_type{T}
  rpd_inform::rpd_inform_type
end

export lpa_initialize

function lpa_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.lpa_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lpa_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function lpa_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.lpa_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lpa_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function lpa_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.lpa_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lpa_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export lpa_read_specfile

function lpa_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.lpa_read_specfile_s(control::Ptr{lpa_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function lpa_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.lpa_read_specfile(control::Ptr{lpa_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function lpa_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.lpa_read_specfile_q(control::Ptr{lpa_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export lpa_import

function lpa_import(::Type{Float32}, control, data, status, n, m, A_type, A_ne, A_row,
                    A_col, A_ptr)
  @ccall libgalahad_single.lpa_import_s(control::Ptr{lpa_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function lpa_import(::Type{Float64}, control, data, status, n, m, A_type, A_ne, A_row,
                    A_col, A_ptr)
  @ccall libgalahad_double.lpa_import(control::Ptr{lpa_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                      A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                      A_ptr::Ptr{Cint})::Cvoid
end

function lpa_import(::Type{Float128}, control, data, status, n, m, A_type, A_ne, A_row,
                    A_col, A_ptr)
  @ccall libgalahad_quadruple.lpa_import_q(control::Ptr{lpa_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                           A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                           A_ptr::Ptr{Cint})::Cvoid
end

export lpa_reset_control

function lpa_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.lpa_reset_control_s(control::Ptr{lpa_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function lpa_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.lpa_reset_control(control::Ptr{lpa_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function lpa_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.lpa_reset_control_q(control::Ptr{lpa_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export lpa_solve_lp

function lpa_solve_lp(::Type{Float32}, data, status, n, m, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.lpa_solve_lp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float32}, f::Float32, a_ne::Cint,
                                          A_val::Ptr{Float32}, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, x::Ptr{Float32},
                                          c::Ptr{Float32}, y::Ptr{Float32}, z::Ptr{Float32},
                                          x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function lpa_solve_lp(::Type{Float64}, data, status, n, m, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.lpa_solve_lp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                        A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                        c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                        x_u::Ptr{Float64}, x::Ptr{Float64}, c::Ptr{Float64},
                                        y::Ptr{Float64}, z::Ptr{Float64}, x_stat::Ptr{Cint},
                                        c_stat::Ptr{Cint})::Cvoid
end

function lpa_solve_lp(::Type{Float128}, data, status, n, m, g, f, a_ne, A_val, c_l, c_u,
                      x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.lpa_solve_lp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, g::Ptr{Float128},
                                             f::Cfloat128, a_ne::Cint, A_val::Ptr{Float128},
                                             c_l::Ptr{Float128}, c_u::Ptr{Float128},
                                             x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                             x::Ptr{Float128}, c::Ptr{Float128},
                                             y::Ptr{Float128}, z::Ptr{Float128},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export lpa_information

function lpa_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.lpa_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{lpa_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function lpa_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.lpa_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lpa_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function lpa_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.lpa_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{lpa_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export lpa_terminate

function lpa_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lpa_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lpa_control_type{Float32}},
                                           inform::Ptr{lpa_inform_type{Float32}})::Cvoid
end

function lpa_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lpa_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lpa_control_type{Float64}},
                                         inform::Ptr{lpa_inform_type{Float64}})::Cvoid
end

function lpa_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.lpa_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{lpa_control_type{Float128}},
                                              inform::Ptr{lpa_inform_type{Float128}})::Cvoid
end
