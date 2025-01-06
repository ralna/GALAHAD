export ugo_control_type

struct ugo_control_type{T,INT}
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  initial_points::INT
  storage_increment::INT
  buffer::INT
  lipschitz_estimate_used::INT
  next_interval_selection::INT
  refine_with_newton::INT
  alive_unit::INT
  alive_file::NTuple{31,Cchar}
  stop_length::T
  small_g_for_newton::T
  small_g::T
  obj_sufficient::T
  global_lipschitz_constant::T
  reliability_parameter::T
  lipschitz_lower_bound::T
  cpu_time_limit::T
  clock_time_limit::T
  second_derivative_available::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export ugo_time_type

struct ugo_time_type{T}
  total::Float32
  clock_total::T
end

export ugo_inform_type

struct ugo_inform_type{T,INT}
  status::INT
  eval_status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  f_eval::INT
  g_eval::INT
  h_eval::INT
  time::ugo_time_type{T}
end

export ugo_initialize

function ugo_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ugo_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function ugo_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ugo_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function ugo_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ugo_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function ugo_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ugo_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function ugo_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ugo_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function ugo_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{ugo_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export ugo_read_specfile

function ugo_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.ugo_read_specfile(control::Ptr{ugo_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.ugo_read_specfile(control::Ptr{ugo_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.ugo_read_specfile(control::Ptr{ugo_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.ugo_read_specfile(control::Ptr{ugo_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.ugo_read_specfile(control::Ptr{ugo_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.ugo_read_specfile(control::Ptr{ugo_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export ugo_import

function ugo_import(::Type{Float32}, ::Type{Int32}, control, data, status, x_l, x_u)
  @ccall libgalahad_single.ugo_import(control::Ptr{ugo_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                      x_l::Ptr{Float32}, x_u::Ptr{Float32})::Cvoid
end

function ugo_import(::Type{Float32}, ::Type{Int64}, control, data, status, x_l, x_u)
  @ccall libgalahad_single_64.ugo_import(control::Ptr{ugo_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         x_l::Ptr{Float32}, x_u::Ptr{Float32})::Cvoid
end

function ugo_import(::Type{Float64}, ::Type{Int32}, control, data, status, x_l, x_u)
  @ccall libgalahad_double.ugo_import(control::Ptr{ugo_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

function ugo_import(::Type{Float64}, ::Type{Int64}, control, data, status, x_l, x_u)
  @ccall libgalahad_double_64.ugo_import(control::Ptr{ugo_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

function ugo_import(::Type{Float128}, ::Type{Int32}, control, data, status, x_l, x_u)
  @ccall libgalahad_quadruple.ugo_import(control::Ptr{ugo_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         x_l::Ptr{Float128}, x_u::Ptr{Float128})::Cvoid
end

function ugo_import(::Type{Float128}, ::Type{Int64}, control, data, status, x_l, x_u)
  @ccall libgalahad_quadruple_64.ugo_import(control::Ptr{ugo_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            x_l::Ptr{Float128}, x_u::Ptr{Float128})::Cvoid
end

export ugo_reset_control

function ugo_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.ugo_reset_control(control::Ptr{ugo_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function ugo_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.ugo_reset_control(control::Ptr{ugo_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function ugo_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.ugo_reset_control(control::Ptr{ugo_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function ugo_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.ugo_reset_control(control::Ptr{ugo_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function ugo_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.ugo_reset_control(control::Ptr{ugo_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function ugo_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.ugo_reset_control(control::Ptr{ugo_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export ugo_solve_direct

function ugo_solve_direct(::Type{Float32}, ::Type{Int32}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_single.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                            status::Ptr{Int32}, x::Ptr{Float32},
                                            f::Ptr{Float32}, g::Ptr{Float32},
                                            h::Ptr{Float32}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float32}, ::Type{Int64}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_single_64.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Int64}, x::Ptr{Float32},
                                               f::Ptr{Float32}, g::Ptr{Float32},
                                               h::Ptr{Float32}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float64}, ::Type{Int32}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_double.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                            status::Ptr{Int32}, x::Ptr{Float64},
                                            f::Ptr{Float64}, g::Ptr{Float64},
                                            h::Ptr{Float64}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float64}, ::Type{Int64}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_double_64.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Int64}, x::Ptr{Float64},
                                               f::Ptr{Float64}, g::Ptr{Float64},
                                               h::Ptr{Float64}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float128}, ::Type{Int32}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_quadruple.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Int32}, x::Ptr{Float128},
                                               f::Ptr{Float128}, g::Ptr{Float128},
                                               h::Ptr{Float128},
                                               eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float128}, ::Type{Int64}, data, userdata, status, x, f, g,
                          h, eval_fgh)
  @ccall libgalahad_quadruple_64.ugo_solve_direct(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                  x::Ptr{Float128}, f::Ptr{Float128},
                                                  g::Ptr{Float128}, h::Ptr{Float128},
                                                  eval_fgh::Ptr{Cvoid})::Cvoid
end

export ugo_solve_reverse

function ugo_solve_reverse(::Type{Float32}, ::Type{Int32}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_single.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             eval_status::Ptr{Int32}, x::Ptr{Float32},
                                             f::Ptr{Float32}, g::Ptr{Float32},
                                             h::Ptr{Float32})::Cvoid
end

function ugo_solve_reverse(::Type{Float32}, ::Type{Int64}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_single_64.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                eval_status::Ptr{Int64}, x::Ptr{Float32},
                                                f::Ptr{Float32}, g::Ptr{Float32},
                                                h::Ptr{Float32})::Cvoid
end

function ugo_solve_reverse(::Type{Float64}, ::Type{Int32}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_double.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             eval_status::Ptr{Int32}, x::Ptr{Float64},
                                             f::Ptr{Float64}, g::Ptr{Float64},
                                             h::Ptr{Float64})::Cvoid
end

function ugo_solve_reverse(::Type{Float64}, ::Type{Int64}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_double_64.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                eval_status::Ptr{Int64}, x::Ptr{Float64},
                                                f::Ptr{Float64}, g::Ptr{Float64},
                                                h::Ptr{Float64})::Cvoid
end

function ugo_solve_reverse(::Type{Float128}, ::Type{Int32}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_quadruple.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                eval_status::Ptr{Int32}, x::Ptr{Float128},
                                                f::Ptr{Float128}, g::Ptr{Float128},
                                                h::Ptr{Float128})::Cvoid
end

function ugo_solve_reverse(::Type{Float128}, ::Type{Int64}, data, status, eval_status, x, f,
                           g, h)
  @ccall libgalahad_quadruple_64.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64},
                                                   eval_status::Ptr{Int64},
                                                   x::Ptr{Float128}, f::Ptr{Float128},
                                                   g::Ptr{Float128},
                                                   h::Ptr{Float128})::Cvoid
end

export ugo_information

function ugo_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.ugo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{ugo_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function ugo_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.ugo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{ugo_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function ugo_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.ugo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{ugo_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function ugo_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.ugo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{ugo_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function ugo_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.ugo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{ugo_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function ugo_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.ugo_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{ugo_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export ugo_terminate

function ugo_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ugo_control_type{Float32,Int32}},
                                         inform::Ptr{ugo_inform_type{Float32,Int32}})::Cvoid
end

function ugo_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ugo_control_type{Float32,Int64}},
                                            inform::Ptr{ugo_inform_type{Float32,Int64}})::Cvoid
end

function ugo_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ugo_control_type{Float64,Int32}},
                                         inform::Ptr{ugo_inform_type{Float64,Int32}})::Cvoid
end

function ugo_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ugo_control_type{Float64,Int64}},
                                            inform::Ptr{ugo_inform_type{Float64,Int64}})::Cvoid
end

function ugo_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ugo_control_type{Float128,Int32}},
                                            inform::Ptr{ugo_inform_type{Float128,Int32}})::Cvoid
end

function ugo_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ugo_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{ugo_inform_type{Float128,Int64}})::Cvoid
end
