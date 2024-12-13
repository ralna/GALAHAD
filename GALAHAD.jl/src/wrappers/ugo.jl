export ugo_control_type

struct ugo_control_type{T}
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  initial_points::Cint
  storage_increment::Cint
  buffer::Cint
  lipschitz_estimate_used::Cint
  next_interval_selection::Cint
  refine_with_newton::Cint
  alive_unit::Cint
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

struct ugo_inform_type{T}
  status::Cint
  eval_status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  f_eval::Cint
  g_eval::Cint
  h_eval::Cint
  time::ugo_time_type{T}
end

export ugo_initialize

function ugo_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.ugo_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ugo_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function ugo_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ugo_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function ugo_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.ugo_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ugo_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export ugo_read_specfile

function ugo_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.ugo_read_specfile_s(control::Ptr{ugo_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.ugo_read_specfile(control::Ptr{ugo_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function ugo_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.ugo_read_specfile_q(control::Ptr{ugo_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export ugo_import

function ugo_import(::Type{Float32}, control, data, status, x_l, x_u)
  @ccall libgalahad_single.ugo_import_s(control::Ptr{ugo_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                        x_l::Ptr{Float32}, x_u::Ptr{Float32})::Cvoid
end

function ugo_import(::Type{Float64}, control, data, status, x_l, x_u)
  @ccall libgalahad_double.ugo_import(control::Ptr{ugo_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

function ugo_import(::Type{Float128}, control, data, status, x_l, x_u)
  @ccall libgalahad_quadruple.ugo_import_q(control::Ptr{ugo_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           x_l::Ptr{Float128}, x_u::Ptr{Float128})::Cvoid
end

export ugo_reset_control

function ugo_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.ugo_reset_control_s(control::Ptr{ugo_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function ugo_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.ugo_reset_control(control::Ptr{ugo_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function ugo_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.ugo_reset_control_q(control::Ptr{ugo_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export ugo_solve_direct

function ugo_solve_direct(::Type{Float32}, data, userdata, status, x, f, g, h, eval_fgh)
  @ccall libgalahad_single.ugo_solve_direct_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, x::Ptr{Float32},
                                              f::Ptr{Float32}, g::Ptr{Float32},
                                              h::Ptr{Float32}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float64}, data, userdata, status, x, f, g, h, eval_fgh)
  @ccall libgalahad_double.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                            status::Ptr{Cint}, x::Ptr{Float64},
                                            f::Ptr{Float64}, g::Ptr{Float64},
                                            h::Ptr{Float64}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_direct(::Type{Float128}, data, userdata, status, x, f, g, h, eval_fgh)
  @ccall libgalahad_quadruple.ugo_solve_direct_q(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                 x::Ptr{Float128}, f::Ptr{Float128},
                                                 g::Ptr{Float128}, h::Ptr{Float128},
                                                 eval_fgh::Ptr{Cvoid})::Cvoid
end

export ugo_solve_reverse

function ugo_solve_reverse(::Type{Float32}, data, status, eval_status, x, f, g, h)
  @ccall libgalahad_single.ugo_solve_reverse_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               eval_status::Ptr{Cint}, x::Ptr{Float32},
                                               f::Ptr{Float32}, g::Ptr{Float32},
                                               h::Ptr{Float32})::Cvoid
end

function ugo_solve_reverse(::Type{Float64}, data, status, eval_status, x, f, g, h)
  @ccall libgalahad_double.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             eval_status::Ptr{Cint}, x::Ptr{Float64},
                                             f::Ptr{Float64}, g::Ptr{Float64},
                                             h::Ptr{Float64})::Cvoid
end

function ugo_solve_reverse(::Type{Float128}, data, status, eval_status, x, f, g, h)
  @ccall libgalahad_quadruple.ugo_solve_reverse_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  eval_status::Ptr{Cint}, x::Ptr{Float128},
                                                  f::Ptr{Float128}, g::Ptr{Float128},
                                                  h::Ptr{Float128})::Cvoid
end

export ugo_information

function ugo_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.ugo_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{ugo_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function ugo_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.ugo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{ugo_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function ugo_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.ugo_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{ugo_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export ugo_terminate

function ugo_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.ugo_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ugo_control_type{Float32}},
                                           inform::Ptr{ugo_inform_type{Float32}})::Cvoid
end

function ugo_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ugo_control_type{Float64}},
                                         inform::Ptr{ugo_inform_type{Float64}})::Cvoid
end

function ugo_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.ugo_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{ugo_control_type{Float128}},
                                              inform::Ptr{ugo_inform_type{Float128}})::Cvoid
end
