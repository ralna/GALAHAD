export dps_control_type

struct dps_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  problem::INT
  print_level::INT
  new_h::INT
  taylor_max_degree::INT
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
  sls_control::sls_control_type{T,INT}
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

struct dps_inform_type{T,INT}
  status::INT
  alloc_status::INT
  mod_1by1::INT
  mod_2by2::INT
  obj::T
  obj_regularized::T
  x_norm::T
  multiplier::T
  pole::T
  hard_case::Bool
  bad_alloc::NTuple{81,Cchar}
  time::dps_time_type{T}
  sls_inform::sls_inform_type{T,INT}
end

export dps_initialize

function dps_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dps_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dps_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dps_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dps_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dps_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dps_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dps_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dps_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dps_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function dps_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{dps_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export dps_read_specfile

function dps_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.dps_read_specfile(control::Ptr{dps_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.dps_read_specfile(control::Ptr{dps_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.dps_read_specfile(control::Ptr{dps_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.dps_read_specfile(control::Ptr{dps_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.dps_read_specfile(control::Ptr{dps_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dps_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.dps_read_specfile(control::Ptr{dps_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export dps_import

function dps_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single.dps_import(control::Ptr{dps_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function dps_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.dps_import(control::Ptr{dps_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function dps_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double.dps_import(control::Ptr{dps_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function dps_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.dps_import(control::Ptr{dps_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function dps_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.dps_import(control::Ptr{dps_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, H_type::Ptr{Cchar}, ne::Int32,
                                         H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32})::Cvoid
end

function dps_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.dps_import(control::Ptr{dps_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                            H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

export dps_reset_control

function dps_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.dps_reset_control(control::Ptr{dps_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dps_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.dps_reset_control(control::Ptr{dps_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dps_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.dps_reset_control(control::Ptr{dps_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dps_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.dps_reset_control(control::Ptr{dps_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dps_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.dps_reset_control(control::Ptr{dps_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function dps_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.dps_reset_control(control::Ptr{dps_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export dps_solve_tr_problem

function dps_solve_tr_problem(::Type{Float32}, ::Type{Int32}, data, status, n, ne, H_val, c,
                              f, radius, x)
  @ccall libgalahad_single.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, ne::Int32, H_val::Ptr{Float32},
                                                c::Ptr{Float32}, f::Float32,
                                                radius::Float32, x::Ptr{Float32})::Cvoid
end

function dps_solve_tr_problem(::Type{Float32}, ::Type{Int64}, data, status, n, ne, H_val, c,
                              f, radius, x)
  @ccall libgalahad_single_64.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, ne::Int64,
                                                   H_val::Ptr{Float32}, c::Ptr{Float32},
                                                   f::Float32, radius::Float32,
                                                   x::Ptr{Float32})::Cvoid
end

function dps_solve_tr_problem(::Type{Float64}, ::Type{Int32}, data, status, n, ne, H_val, c,
                              f, radius, x)
  @ccall libgalahad_double.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, ne::Int32, H_val::Ptr{Float64},
                                                c::Ptr{Float64}, f::Float64,
                                                radius::Float64, x::Ptr{Float64})::Cvoid
end

function dps_solve_tr_problem(::Type{Float64}, ::Type{Int64}, data, status, n, ne, H_val, c,
                              f, radius, x)
  @ccall libgalahad_double_64.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, ne::Int64,
                                                   H_val::Ptr{Float64}, c::Ptr{Float64},
                                                   f::Float64, radius::Float64,
                                                   x::Ptr{Float64})::Cvoid
end

function dps_solve_tr_problem(::Type{Float128}, ::Type{Int32}, data, status, n, ne, H_val,
                              c, f, radius, x)
  @ccall libgalahad_quadruple.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32, ne::Int32,
                                                   H_val::Ptr{Float128}, c::Ptr{Float128},
                                                   f::Cfloat128, radius::Cfloat128,
                                                   x::Ptr{Float128})::Cvoid
end

function dps_solve_tr_problem(::Type{Float128}, ::Type{Int64}, data, status, n, ne, H_val,
                              c, f, radius, x)
  @ccall libgalahad_quadruple_64.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64,
                                                      ne::Int64, H_val::Ptr{Float128},
                                                      c::Ptr{Float128}, f::Cfloat128,
                                                      radius::Cfloat128,
                                                      x::Ptr{Float128})::Cvoid
end

export dps_solve_rq_problem

function dps_solve_rq_problem(::Type{Float32}, ::Type{Int32}, data, status, n, ne, H_val, c,
                              f, power, weight, x)
  @ccall libgalahad_single.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, ne::Int32, H_val::Ptr{Float32},
                                                c::Ptr{Float32}, f::Float32, power::Float32,
                                                weight::Float32, x::Ptr{Float32})::Cvoid
end

function dps_solve_rq_problem(::Type{Float32}, ::Type{Int64}, data, status, n, ne, H_val, c,
                              f, power, weight, x)
  @ccall libgalahad_single_64.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, ne::Int64,
                                                   H_val::Ptr{Float32}, c::Ptr{Float32},
                                                   f::Float32, power::Float32,
                                                   weight::Float32, x::Ptr{Float32})::Cvoid
end

function dps_solve_rq_problem(::Type{Float64}, ::Type{Int32}, data, status, n, ne, H_val, c,
                              f, power, weight, x)
  @ccall libgalahad_double.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, ne::Int32, H_val::Ptr{Float64},
                                                c::Ptr{Float64}, f::Float64, power::Float64,
                                                weight::Float64, x::Ptr{Float64})::Cvoid
end

function dps_solve_rq_problem(::Type{Float64}, ::Type{Int64}, data, status, n, ne, H_val, c,
                              f, power, weight, x)
  @ccall libgalahad_double_64.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, ne::Int64,
                                                   H_val::Ptr{Float64}, c::Ptr{Float64},
                                                   f::Float64, power::Float64,
                                                   weight::Float64, x::Ptr{Float64})::Cvoid
end

function dps_solve_rq_problem(::Type{Float128}, ::Type{Int32}, data, status, n, ne, H_val,
                              c, f, power, weight, x)
  @ccall libgalahad_quadruple.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32, ne::Int32,
                                                   H_val::Ptr{Float128}, c::Ptr{Float128},
                                                   f::Cfloat128, power::Cfloat128,
                                                   weight::Cfloat128,
                                                   x::Ptr{Float128})::Cvoid
end

function dps_solve_rq_problem(::Type{Float128}, ::Type{Int64}, data, status, n, ne, H_val,
                              c, f, power, weight, x)
  @ccall libgalahad_quadruple_64.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64,
                                                      ne::Int64, H_val::Ptr{Float128},
                                                      c::Ptr{Float128}, f::Cfloat128,
                                                      power::Cfloat128, weight::Cfloat128,
                                                      x::Ptr{Float128})::Cvoid
end

export dps_resolve_tr_problem

function dps_resolve_tr_problem(::Type{Float32}, ::Type{Int32}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_single.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                  n::Int32, c::Ptr{Float32}, f::Float32,
                                                  radius::Float32, x::Ptr{Float32})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float32}, ::Type{Int64}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_single_64.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64}, n::Int64,
                                                     c::Ptr{Float32}, f::Float32,
                                                     radius::Float32,
                                                     x::Ptr{Float32})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float64}, ::Type{Int32}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_double.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                  n::Int32, c::Ptr{Float64}, f::Float64,
                                                  radius::Float64, x::Ptr{Float64})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float64}, ::Type{Int64}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_double_64.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64}, n::Int64,
                                                     c::Ptr{Float64}, f::Float64,
                                                     radius::Float64,
                                                     x::Ptr{Float64})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float128}, ::Type{Int32}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_quadruple.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n::Int32,
                                                     c::Ptr{Float128}, f::Cfloat128,
                                                     radius::Cfloat128,
                                                     x::Ptr{Float128})::Cvoid
end

function dps_resolve_tr_problem(::Type{Float128}, ::Type{Int64}, data, status, n, c, f,
                                radius, x)
  @ccall libgalahad_quadruple_64.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n::Int64,
                                                        c::Ptr{Float128}, f::Cfloat128,
                                                        radius::Cfloat128,
                                                        x::Ptr{Float128})::Cvoid
end

export dps_resolve_rq_problem

function dps_resolve_rq_problem(::Type{Float32}, ::Type{Int32}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_single.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                  n::Int32, c::Ptr{Float32}, f::Float32,
                                                  power::Float32, weight::Float32,
                                                  x::Ptr{Float32})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float32}, ::Type{Int64}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_single_64.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64}, n::Int64,
                                                     c::Ptr{Float32}, f::Float32,
                                                     power::Float32, weight::Float32,
                                                     x::Ptr{Float32})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float64}, ::Type{Int32}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_double.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                  n::Int32, c::Ptr{Float64}, f::Float64,
                                                  power::Float64, weight::Float64,
                                                  x::Ptr{Float64})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float64}, ::Type{Int64}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_double_64.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64}, n::Int64,
                                                     c::Ptr{Float64}, f::Float64,
                                                     power::Float64, weight::Float64,
                                                     x::Ptr{Float64})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float128}, ::Type{Int32}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_quadruple.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n::Int32,
                                                     c::Ptr{Float128}, f::Cfloat128,
                                                     power::Cfloat128, weight::Cfloat128,
                                                     x::Ptr{Float128})::Cvoid
end

function dps_resolve_rq_problem(::Type{Float128}, ::Type{Int64}, data, status, n, c, f,
                                power, weight, x)
  @ccall libgalahad_quadruple_64.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n::Int64,
                                                        c::Ptr{Float128}, f::Cfloat128,
                                                        power::Cfloat128, weight::Cfloat128,
                                                        x::Ptr{Float128})::Cvoid
end

export dps_information

function dps_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.dps_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dps_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dps_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.dps_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dps_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dps_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.dps_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dps_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dps_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.dps_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dps_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dps_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.dps_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dps_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function dps_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.dps_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{dps_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export dps_terminate

function dps_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dps_control_type{Float32,Int32}},
                                         inform::Ptr{dps_inform_type{Float32,Int32}})::Cvoid
end

function dps_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dps_control_type{Float32,Int64}},
                                            inform::Ptr{dps_inform_type{Float32,Int64}})::Cvoid
end

function dps_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dps_control_type{Float64,Int32}},
                                         inform::Ptr{dps_inform_type{Float64,Int32}})::Cvoid
end

function dps_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dps_control_type{Float64,Int64}},
                                            inform::Ptr{dps_inform_type{Float64,Int64}})::Cvoid
end

function dps_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dps_control_type{Float128,Int32}},
                                            inform::Ptr{dps_inform_type{Float128,Int32}})::Cvoid
end

function dps_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{dps_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{dps_inform_type{Float128,Int64}})::Cvoid
end
