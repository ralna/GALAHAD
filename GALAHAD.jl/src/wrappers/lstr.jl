export lstr_control_type

mutable struct lstr_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  itmin::Cint
  itmax::Cint
  itmax_on_boundary::Cint
  bitmax::Cint
  extra_vectors::Cint
  stop_relative::T
  stop_absolute::T
  fraction_opt::T
  time_limit::T
  steihaug_toint::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}

  lstr_control_type{T}() where T = new()
end

export lstr_inform_type

mutable struct lstr_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  iter_pass2::Cint
  biters::Cint
  biter_min::Cint
  biter_max::Cint
  multiplier::T
  x_norm::T
  r_norm::T
  Atr_norm::T
  biter_mean::T

  lstr_inform_type{T}() where T = new()
end

export lstr_initialize_s

function lstr_initialize_s(data, control, status)
  @ccall libgalahad_single.lstr_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{lstr_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export lstr_initialize

function lstr_initialize(data, control, status)
  @ccall libgalahad_double.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{lstr_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export lstr_read_specfile_s

function lstr_read_specfile_s(control, specfile)
  @ccall libgalahad_single.lstr_read_specfile_s(control::Ref{lstr_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export lstr_read_specfile

function lstr_read_specfile(control, specfile)
  @ccall libgalahad_double.lstr_read_specfile(control::Ref{lstr_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export lstr_import_control_s

function lstr_import_control_s(control, data, status)
  @ccall libgalahad_single.lstr_import_control_s(control::Ref{lstr_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

export lstr_import_control

function lstr_import_control(control, data, status)
  @ccall libgalahad_double.lstr_import_control(control::Ref{lstr_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export lstr_solve_problem_s

function lstr_solve_problem_s(data, status, m, n, radius, x, u, v)
  @ccall libgalahad_single.lstr_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, radius::Float32,
                                                x::Ptr{Float32}, u::Ptr{Float32},
                                                v::Ptr{Float32})::Cvoid
end

export lstr_solve_problem

function lstr_solve_problem(data, status, m, n, radius, x, u, v)
  @ccall libgalahad_double.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, radius::Float64,
                                              x::Ptr{Float64}, u::Ptr{Float64},
                                              v::Ptr{Float64})::Cvoid
end

export lstr_information_s

function lstr_information_s(data, inform, status)
  @ccall libgalahad_single.lstr_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{lstr_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export lstr_information

function lstr_information(data, inform, status)
  @ccall libgalahad_double.lstr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{lstr_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export lstr_terminate_s

function lstr_terminate_s(data, control, inform)
  @ccall libgalahad_single.lstr_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{lstr_control_type{Float32}},
                                            inform::Ref{lstr_inform_type{Float32}})::Cvoid
end

export lstr_terminate

function lstr_terminate(data, control, inform)
  @ccall libgalahad_double.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{lstr_control_type{Float64}},
                                          inform::Ref{lstr_inform_type{Float64}})::Cvoid
end
