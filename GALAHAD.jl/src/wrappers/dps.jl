export dps_control_type

struct dps_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  problem::Cint
  print_level::Cint
  new_h::Cint
  taylor_max_degree::Cint
  eigen_min::T
  lower::T
  upper::T
  stop_normal::T
  stop_absolute_normal::T
  goldfarb::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  problem_file::NTuple{31,Cchar}
  symmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
end

export dps_time_type

struct dps_time_type{T}
  total::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export dps_inform_type

struct dps_inform_type{T}
  status::Cint
  alloc_status::Cint
  mod_1by1::Cint
  mod_2by2::Cint
  obj::T
  obj_regularized::T
  x_norm::T
  multiplier::T
  pole::T
  hard_case::Bool
  bad_alloc::NTuple{81,Cchar}
  time::dps_time_type{T}
  sls_inform::sls_inform_type{T}
end

export dps_initialize

function dps_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.dps_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dps_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function dps_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dps_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export dps_read_specfile

function dps_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.dps_read_specfile_s(control::Ptr{dps_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.dps_read_specfile(control::Ptr{dps_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export dps_import

function dps_import(::Type{Float32}, control, data, status, n, H_type, ne, H_row, H_col,
                    H_ptr)
  @ccall libgalahad_single.dps_import_s(control::Ptr{dps_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function dps_import(::Type{Float64}, control, data, status, n, H_type, ne, H_row, H_col,
                    H_ptr)
  @ccall libgalahad_double.dps_import(control::Ptr{dps_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export dps_reset_control

function dps_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.dps_reset_control_s(control::Ptr{dps_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function dps_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.dps_reset_control(control::Ptr{dps_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export dps_solve_tr_problem

function dps_solve_tr_problem(::Type{Float32}, data, status, n, ne, H_val, c, f, radius, x)
  @ccall libgalahad_single.dps_solve_tr_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, ne::Cint, H_val::Ptr{Float32},
                                                  c::Ptr{Float32}, f::Float32,
                                                  radius::Float32, x::Ptr{Float32})::Cvoid
end

function dps_solve_tr_problem(::Type{Float64}, data, status, n, ne, H_val, c, f, radius, x)
  @ccall libgalahad_double.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, ne::Cint, H_val::Ptr{Float64},
                                                c::Ptr{Float64}, f::Float64,
                                                radius::Float64, x::Ptr{Float64})::Cvoid
end

export dps_solve_rq_problem

function dps_solve_rq_problem(::Type{Float32}, data, status, n, ne, H_val, c, f, power,
                              weight, x)
  @ccall libgalahad_single.dps_solve_rq_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, ne::Cint, H_val::Ptr{Float32},
                                                  c::Ptr{Float32}, f::Float32,
                                                  power::Float32, weight::Float32,
                                                  x::Ptr{Float32})::Cvoid
end

function dps_solve_rq_problem(::Type{Float64}, data, status, n, ne, H_val, c, f, power,
                              weight, x)
  @ccall libgalahad_double.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, ne::Cint, H_val::Ptr{Float64},
                                                c::Ptr{Float64}, f::Float64, power::Float64,
                                                weight::Float64, x::Ptr{Float64})::Cvoid
end

export dps_resolve_tr_problem

function dps_resolve_tr_problem(::Type{Float32}, data, status, n, c, f, radius, x)
  @ccall libgalahad_single.dps_resolve_tr_problem_s(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    c::Ptr{Float32}, f::Float32,
                                                    radius::Float32, x::Ptr{Float32})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float64}, data, status, n, c, f, radius, x)
  @ccall libgalahad_double.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, c::Ptr{Float64}, f::Float64,
                                                  radius::Float64, x::Ptr{Float64})::Cvoid
end

export dps_resolve_rq_problem

function dps_resolve_rq_problem(::Type{Float32}, data, status, n, c, f, power, weight, x)
  @ccall libgalahad_single.dps_resolve_rq_problem_s(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    c::Ptr{Float32}, f::Float32,
                                                    power::Float32, weight::Float32,
                                                    x::Ptr{Float32})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float64}, data, status, n, c, f, power, weight, x)
  @ccall libgalahad_double.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, c::Ptr{Float64}, f::Float64,
                                                  power::Float64, weight::Float64,
                                                  x::Ptr{Float64})::Cvoid
end

export dps_information

function dps_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.dps_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{dps_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function dps_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.dps_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dps_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export dps_terminate

function dps_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.dps_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{dps_control_type{Float32}},
                                           inform::Ptr{dps_inform_type{Float32}})::Cvoid
end

function dps_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dps_control_type{Float64}},
                                         inform::Ptr{dps_inform_type{Float64}})::Cvoid
end
