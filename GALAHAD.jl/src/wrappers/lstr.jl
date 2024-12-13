export lstr_control_type

struct lstr_control_type{T}
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
end

export lstr_inform_type

struct lstr_inform_type{T}
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
end

export lstr_initialize

function lstr_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.lstr_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lstr_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function lstr_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lstr_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function lstr_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.lstr_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{lstr_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export lstr_read_specfile

function lstr_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.lstr_read_specfile_s(control::Ptr{lstr_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.lstr_read_specfile(control::Ptr{lstr_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.lstr_read_specfile_q(control::Ptr{lstr_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export lstr_import_control

function lstr_import_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.lstr_import_control_s(control::Ptr{lstr_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function lstr_import_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.lstr_import_control(control::Ptr{lstr_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function lstr_import_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.lstr_import_control_q(control::Ptr{lstr_control_type{Float128}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint})::Cvoid
end

export lstr_solve_problem

function lstr_solve_problem(::Type{Float32}, data, status, m, n, radius, x, u, v)
  @ccall libgalahad_single.lstr_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, radius::Float32,
                                                x::Ptr{Float32}, u::Ptr{Float32},
                                                v::Ptr{Float32})::Cvoid
end

function lstr_solve_problem(::Type{Float64}, data, status, m, n, radius, x, u, v)
  @ccall libgalahad_double.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, radius::Float64,
                                              x::Ptr{Float64}, u::Ptr{Float64},
                                              v::Ptr{Float64})::Cvoid
end

function lstr_solve_problem(::Type{Float128}, data, status, m, n, radius, x, u, v)
  @ccall libgalahad_quadruple.lstr_solve_problem_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   m::Cint, n::Cint, radius::Float128,
                                                   x::Ptr{Float128}, u::Ptr{Float128},
                                                   v::Ptr{Float128})::Cvoid
end

export lstr_information

function lstr_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.lstr_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lstr_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function lstr_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.lstr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{lstr_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function lstr_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.lstr_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{lstr_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export lstr_terminate

function lstr_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lstr_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lstr_control_type{Float32}},
                                            inform::Ptr{lstr_inform_type{Float32}})::Cvoid
end

function lstr_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lstr_control_type{Float64}},
                                          inform::Ptr{lstr_inform_type{Float64}})::Cvoid
end

function lstr_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.lstr_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lstr_control_type{Float128}},
                                               inform::Ptr{lstr_inform_type{Float128}})::Cvoid
end
