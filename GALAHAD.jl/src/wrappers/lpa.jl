export lpa_control_type

mutable struct lpa_control_type{T}
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

  lpa_control_type{T}() where T = new()
end

export lpa_time_type

mutable struct lpa_time_type{T}
  total::T
  preprocess::T
  clock_total::T
  clock_preprocess::T

  lpa_time_type{T}() where T = new()
end

export lpa_inform_type

mutable struct lpa_inform_type{T}
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

  lpa_inform_type{T}() where T = new()
end

export lpa_initialize_s

function lpa_initialize_s(data, control, status)
  @ccall libgalahad_single.lpa_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{lpa_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export lpa_initialize

function lpa_initialize(data, control, status)
  @ccall libgalahad_double.lpa_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{lpa_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export lpa_read_specfile_s

function lpa_read_specfile_s(control, specfile)
  @ccall libgalahad_single.lpa_read_specfile_s(control::Ref{lpa_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export lpa_read_specfile

function lpa_read_specfile(control, specfile)
  @ccall libgalahad_double.lpa_read_specfile(control::Ref{lpa_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export lpa_import_s

function lpa_import_s(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.lpa_import_s(control::Ref{lpa_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

export lpa_import

function lpa_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.lpa_import(control::Ref{lpa_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                      A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                      A_ptr::Ptr{Cint})::Cvoid
end

export lpa_reset_control_s

function lpa_reset_control_s(control, data, status)
  @ccall libgalahad_single.lpa_reset_control_s(control::Ref{lpa_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export lpa_reset_control

function lpa_reset_control(control, data, status)
  @ccall libgalahad_double.lpa_reset_control(control::Ref{lpa_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export lpa_solve_lp_s

function lpa_solve_lp_s(data, status, n, m, g, f, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                      x_stat, c_stat)
  @ccall libgalahad_single.lpa_solve_lp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float32}, f::Float32,
                                          a_ne::Cint, A_val::Ptr{Float32},
                                          c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                          x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                          x::Ptr{Float32}, c::Ptr{Float32},
                                          y::Ptr{Float32}, z::Ptr{Float32},
                                          x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export lpa_solve_lp

function lpa_solve_lp(data, status, n, m, g, f, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
  @ccall libgalahad_double.lpa_solve_lp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, g::Ptr{Float64}, f::Float64,
                                        a_ne::Cint, A_val::Ptr{Float64},
                                        c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                        x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                        x::Ptr{Float64}, c::Ptr{Float64},
                                        y::Ptr{Float64}, z::Ptr{Float64},
                                        x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

export lpa_information_s

function lpa_information_s(data, inform, status)
  @ccall libgalahad_single.lpa_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{lpa_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export lpa_information

function lpa_information(data, inform, status)
  @ccall libgalahad_double.lpa_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{lpa_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export lpa_terminate_s

function lpa_terminate_s(data, control, inform)
  @ccall libgalahad_single.lpa_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{lpa_control_type{Float32}},
                                           inform::Ref{lpa_inform_type{Float32}})::Cvoid
end

export lpa_terminate

function lpa_terminate(data, control, inform)
  @ccall libgalahad_double.lpa_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{lpa_control_type{Float64}},
                                         inform::Ref{lpa_inform_type{Float64}})::Cvoid
end
