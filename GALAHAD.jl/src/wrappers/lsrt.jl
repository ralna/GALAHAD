export lsrt_control_type

struct lsrt_control_type{T}
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
end

export lsrt_inform_type

struct lsrt_inform_type{T}
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
end

export lsrt_initialize

function lsrt_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.lsrt_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lsrt_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function lsrt_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.lsrt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lsrt_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function lsrt_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.lsrt_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{lsrt_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export lsrt_read_specfile

function lsrt_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.lsrt_read_specfile_s(control::Ptr{lsrt_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lsrt_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.lsrt_read_specfile(control::Ptr{lsrt_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function lsrt_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.lsrt_read_specfile_q(control::Ptr{lsrt_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export lsrt_import_control

function lsrt_import_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.lsrt_import_control_s(control::Ptr{lsrt_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function lsrt_import_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.lsrt_import_control(control::Ptr{lsrt_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function lsrt_import_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.lsrt_import_control_q(control::Ptr{lsrt_control_type{Float128}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint})::Cvoid
end

export lsrt_solve_problem

function lsrt_solve_problem(::Type{Float32}, data, status, m, n, power, weight, x, u, v)
  @ccall libgalahad_single.lsrt_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float32,
                                                weight::Float32, x::Ptr{Float32},
                                                u::Ptr{Float32}, v::Ptr{Float32})::Cvoid
end

function lsrt_solve_problem(::Type{Float64}, data, status, m, n, power, weight, x, u, v)
  @ccall libgalahad_double.lsrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, power::Float64,
                                              weight::Float64, x::Ptr{Float64},
                                              u::Ptr{Float64}, v::Ptr{Float64})::Cvoid
end

function lsrt_solve_problem(::Type{Float128}, data, status, m, n, power, weight, x, u, v)
  @ccall libgalahad_quadruple.lsrt_solve_problem_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   m::Cint, n::Cint, power::Cfloat128,
                                                   weight::Cfloat128, x::Ptr{Float128},
                                                   u::Ptr{Float128},
                                                   v::Ptr{Float128})::Cvoid
end

export lsrt_information

function lsrt_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.lsrt_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lsrt_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function lsrt_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.lsrt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{lsrt_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function lsrt_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.lsrt_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{lsrt_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export lsrt_terminate

function lsrt_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lsrt_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lsrt_control_type{Float32}},
                                            inform::Ptr{lsrt_inform_type{Float32}})::Cvoid
end

function lsrt_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lsrt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lsrt_control_type{Float64}},
                                          inform::Ptr{lsrt_inform_type{Float64}})::Cvoid
end

function lsrt_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.lsrt_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lsrt_control_type{Float128}},
                                               inform::Ptr{lsrt_inform_type{Float128}})::Cvoid
end
