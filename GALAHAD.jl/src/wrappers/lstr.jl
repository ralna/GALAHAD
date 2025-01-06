export lstr_control_type

struct lstr_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  itmin::INT
  itmax::INT
  itmax_on_boundary::INT
  bitmax::INT
  extra_vectors::INT
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

struct lstr_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  iter_pass2::INT
  biters::INT
  biter_min::INT
  biter_max::INT
  multiplier::T
  x_norm::T
  r_norm::T
  Atr_norm::T
  biter_mean::T
end

export lstr_initialize

function lstr_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lstr_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lstr_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{lstr_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lstr_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lstr_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lstr_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{lstr_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lstr_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{lstr_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function lstr_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{lstr_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export lstr_read_specfile

function lstr_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.lstr_read_specfile(control::Ptr{lstr_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.lstr_read_specfile(control::Ptr{lstr_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.lstr_read_specfile(control::Ptr{lstr_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.lstr_read_specfile(control::Ptr{lstr_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.lstr_read_specfile(control::Ptr{lstr_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function lstr_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.lstr_read_specfile(control::Ptr{lstr_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export lstr_import_control

function lstr_import_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.lstr_import_control(control::Ptr{lstr_control_type{Float32,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function lstr_import_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.lstr_import_control(control::Ptr{lstr_control_type{Float32,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function lstr_import_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.lstr_import_control(control::Ptr{lstr_control_type{Float64,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function lstr_import_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.lstr_import_control(control::Ptr{lstr_control_type{Float64,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function lstr_import_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.lstr_import_control(control::Ptr{lstr_control_type{Float128,
                                                                                 Int32}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int32})::Cvoid
end

function lstr_import_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.lstr_import_control(control::Ptr{lstr_control_type{Float128,
                                                                                    Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

export lstr_solve_problem

function lstr_solve_problem(::Type{Float32}, ::Type{Int32}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_single.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              m::Int32, n::Int32, radius::Float32,
                                              x::Ptr{Float32}, u::Ptr{Float32},
                                              v::Ptr{Float32})::Cvoid
end

function lstr_solve_problem(::Type{Float32}, ::Type{Int64}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_single_64.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 m::Int64, n::Int64, radius::Float32,
                                                 x::Ptr{Float32}, u::Ptr{Float32},
                                                 v::Ptr{Float32})::Cvoid
end

function lstr_solve_problem(::Type{Float64}, ::Type{Int32}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_double.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              m::Int32, n::Int32, radius::Float64,
                                              x::Ptr{Float64}, u::Ptr{Float64},
                                              v::Ptr{Float64})::Cvoid
end

function lstr_solve_problem(::Type{Float64}, ::Type{Int64}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_double_64.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 m::Int64, n::Int64, radius::Float64,
                                                 x::Ptr{Float64}, u::Ptr{Float64},
                                                 v::Ptr{Float64})::Cvoid
end

function lstr_solve_problem(::Type{Float128}, ::Type{Int32}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_quadruple.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 m::Int32, n::Int32, radius::Cfloat128,
                                                 x::Ptr{Float128}, u::Ptr{Float128},
                                                 v::Ptr{Float128})::Cvoid
end

function lstr_solve_problem(::Type{Float128}, ::Type{Int64}, data, status, m, n, radius, x,
                            u, v)
  @ccall libgalahad_quadruple_64.lstr_solve_problem(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, m::Int64, n::Int64,
                                                    radius::Cfloat128, x::Ptr{Float128},
                                                    u::Ptr{Float128},
                                                    v::Ptr{Float128})::Cvoid
end

export lstr_information

function lstr_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.lstr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{lstr_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function lstr_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.lstr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{lstr_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function lstr_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.lstr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{lstr_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function lstr_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.lstr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{lstr_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function lstr_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.lstr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{lstr_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function lstr_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.lstr_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{lstr_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export lstr_terminate

function lstr_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lstr_control_type{Float32,Int32}},
                                          inform::Ptr{lstr_inform_type{Float32,Int32}})::Cvoid
end

function lstr_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lstr_control_type{Float32,Int64}},
                                             inform::Ptr{lstr_inform_type{Float32,Int64}})::Cvoid
end

function lstr_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lstr_control_type{Float64,Int32}},
                                          inform::Ptr{lstr_inform_type{Float64,Int32}})::Cvoid
end

function lstr_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lstr_control_type{Float64,Int64}},
                                             inform::Ptr{lstr_inform_type{Float64,Int64}})::Cvoid
end

function lstr_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lstr_control_type{Float128,Int32}},
                                             inform::Ptr{lstr_inform_type{Float128,Int32}})::Cvoid
end

function lstr_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{lstr_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{lstr_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
