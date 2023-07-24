export lsrt_control_type

mutable struct lsrt_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  itmin::Cint
  itmax::Cint
  bitmax::Cint
  extra_vectors::Cint
  stopping_rule::Cint
  freq::Cint
  stop_relative::T
  stop_absolute::T
  fraction_opt::T
  time_limit::T
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}

  lsrt_control_type{T}() where T = new()
end

export lsrt_inform_type

mutable struct lsrt_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  iter_pass2::Cint
  biters::Cint
  biter_min::Cint
  biter_max::Cint
  obj::T
  multiplier::T
  x_norm::T
  r_norm::T
  Atr_norm::T
  biter_mean::T

  lsrt_inform_type{T}() where T = new()
end

export lsrt_initialize_s

function lsrt_initialize_s(data, control, status)
  @ccall libgalahad_single.lsrt_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{lsrt_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export lsrt_initialize

function lsrt_initialize(data, control, status)
  @ccall libgalahad_double.lsrt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{lsrt_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export lsrt_read_specfile_s

function lsrt_read_specfile_s(control, specfile)
  @ccall libgalahad_single.lsrt_read_specfile_s(control::Ref{lsrt_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export lsrt_read_specfile

function lsrt_read_specfile(control, specfile)
  @ccall libgalahad_double.lsrt_read_specfile(control::Ref{lsrt_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export lsrt_import_control_s

function lsrt_import_control_s(control, data, status)
  @ccall libgalahad_single.lsrt_import_control_s(control::Ref{lsrt_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

export lsrt_import_control

function lsrt_import_control(control, data, status)
  @ccall libgalahad_double.lsrt_import_control(control::Ref{lsrt_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export lsrt_solve_problem_s

function lsrt_solve_problem_s(data, status, m, n, power, weight, x, u, v)
  @ccall libgalahad_single.lsrt_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float32,
                                                weight::Float32, x::Ptr{Float32},
                                                u::Ptr{Float32}, v::Ptr{Float32})::Cvoid
end

export lsrt_solve_problem

function lsrt_solve_problem(data, status, m, n, power, weight, x, u, v)
  @ccall libgalahad_double.lsrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, power::Float64,
                                              weight::Float64, x::Ptr{Float64},
                                              u::Ptr{Float64}, v::Ptr{Float64})::Cvoid
end

export lsrt_information_s

function lsrt_information_s(data, inform, status)
  @ccall libgalahad_single.lsrt_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{lsrt_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export lsrt_information

function lsrt_information(data, inform, status)
  @ccall libgalahad_double.lsrt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{lsrt_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export lsrt_terminate_s

function lsrt_terminate_s(data, control, inform)
  @ccall libgalahad_single.lsrt_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{lsrt_control_type{Float32}},
                                            inform::Ref{lsrt_inform_type{Float32}})::Cvoid
end

export lsrt_terminate

function lsrt_terminate(data, control, inform)
  @ccall libgalahad_double.lsrt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{lsrt_control_type{Float64}},
                                          inform::Ref{lsrt_inform_type{Float64}})::Cvoid
end
