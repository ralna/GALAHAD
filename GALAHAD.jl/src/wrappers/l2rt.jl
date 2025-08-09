export l2rt_control_type

struct l2rt_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  itmin::INT
  itmax::INT
  bitmax::INT
  extra_vectors::INT
  stopping_rule::INT
  freq::INT
  stop_relative::T
  stop_absolute::T
  fraction_opt::T
  time_limit::T
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export l2rt_inform_type

struct l2rt_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  iter_pass2::INT
  biters::INT
  biter_min::INT
  biter_max::INT
  obj::T
  multiplier::T
  x_norm::T
  r_norm::T
  Atr_norm::T
  biter_mean::T
end

export l2rt_initialize

function l2rt_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.l2rt_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{l2rt_control_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function l2rt_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.l2rt_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{l2rt_control_type{Float32,
                                                                                  Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function l2rt_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.l2rt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{l2rt_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function l2rt_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.l2rt_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{l2rt_control_type{Float64,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function l2rt_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.l2rt_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{l2rt_control_type{Float128,
                                                                               Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function l2rt_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.l2rt_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{l2rt_control_type{Float128,
                                                                                     Int64}},
                                                      status::Ptr{Int64})::Cvoid
end

export l2rt_read_specfile

function l2rt_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.l2rt_read_specfile_s(control::Ptr{l2rt_control_type{Float32,
                                                                               Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function l2rt_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.l2rt_read_specfile_s_64(control::Ptr{l2rt_control_type{Float32,
                                                                                     Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

function l2rt_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.l2rt_read_specfile(control::Ptr{l2rt_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function l2rt_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.l2rt_read_specfile_64(control::Ptr{l2rt_control_type{Float64,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function l2rt_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.l2rt_read_specfile_q(control::Ptr{l2rt_control_type{Float128,
                                                                                  Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function l2rt_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.l2rt_read_specfile_q_64(control::Ptr{l2rt_control_type{Float128,
                                                                                        Int64}},
                                                         specfile::Ptr{Cchar})::Cvoid
end

export l2rt_import_control

function l2rt_import_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.l2rt_import_control_s(control::Ptr{l2rt_control_type{Float32,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function l2rt_import_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.l2rt_import_control_s_64(control::Ptr{l2rt_control_type{Float32,
                                                                                      Int64}},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64})::Cvoid
end

function l2rt_import_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.l2rt_import_control(control::Ptr{l2rt_control_type{Float64,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function l2rt_import_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.l2rt_import_control_64(control::Ptr{l2rt_control_type{Float64,
                                                                                    Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

function l2rt_import_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.l2rt_import_control_q(control::Ptr{l2rt_control_type{Float128,
                                                                                   Int32}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32})::Cvoid
end

function l2rt_import_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.l2rt_import_control_q_64(control::Ptr{l2rt_control_type{Float128,
                                                                                         Int64}},
                                                          data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64})::Cvoid
end

export l2rt_solve_problem

function l2rt_solve_problem(::Type{Float32}, ::Type{Int32}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_single.l2rt_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                m::Int32, n::Int32, power::Float32,
                                                weight::Float32, shift::Float32,
                                                x::Ptr{Float32}, u::Ptr{Float32},
                                                v::Ptr{Float32})::Cvoid
end

function l2rt_solve_problem(::Type{Float32}, ::Type{Int64}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_single_64.l2rt_solve_problem_s_64(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, m::Int64,
                                                      n::Int64, power::Float32,
                                                      weight::Float32, shift::Float32,
                                                      x::Ptr{Float32}, u::Ptr{Float32},
                                                      v::Ptr{Float32})::Cvoid
end

function l2rt_solve_problem(::Type{Float64}, ::Type{Int32}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_double.l2rt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              m::Int32, n::Int32, power::Float64,
                                              weight::Float64, shift::Float64,
                                              x::Ptr{Float64}, u::Ptr{Float64},
                                              v::Ptr{Float64})::Cvoid
end

function l2rt_solve_problem(::Type{Float64}, ::Type{Int64}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_double_64.l2rt_solve_problem_64(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, m::Int64, n::Int64,
                                                    power::Float64, weight::Float64,
                                                    shift::Float64, x::Ptr{Float64},
                                                    u::Ptr{Float64}, v::Ptr{Float64})::Cvoid
end

function l2rt_solve_problem(::Type{Float128}, ::Type{Int32}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_quadruple.l2rt_solve_problem_q(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, m::Int32, n::Int32,
                                                   power::Cfloat128, weight::Cfloat128,
                                                   shift::Cfloat128, x::Ptr{Float128},
                                                   u::Ptr{Float128},
                                                   v::Ptr{Float128})::Cvoid
end

function l2rt_solve_problem(::Type{Float128}, ::Type{Int64}, data, status, m, n, power,
                            weight, shift, x, u, v)
  @ccall libgalahad_quadruple_64.l2rt_solve_problem_q_64(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, m::Int64,
                                                         n::Int64, power::Cfloat128,
                                                         weight::Cfloat128,
                                                         shift::Cfloat128, x::Ptr{Float128},
                                                         u::Ptr{Float128},
                                                         v::Ptr{Float128})::Cvoid
end

export l2rt_information

function l2rt_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.l2rt_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{l2rt_inform_type{Float32,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function l2rt_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.l2rt_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{l2rt_inform_type{Float32,
                                                                                 Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function l2rt_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.l2rt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{l2rt_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function l2rt_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.l2rt_information_64(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{l2rt_inform_type{Float64,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function l2rt_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.l2rt_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{l2rt_inform_type{Float128,
                                                                              Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function l2rt_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.l2rt_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                       inform::Ptr{l2rt_inform_type{Float128,
                                                                                    Int64}},
                                                       status::Ptr{Int64})::Cvoid
end

export l2rt_terminate

function l2rt_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.l2rt_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{l2rt_control_type{Float32,Int32}},
                                            inform::Ptr{l2rt_inform_type{Float32,Int32}})::Cvoid
end

function l2rt_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.l2rt_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{l2rt_control_type{Float32,
                                                                                 Int64}},
                                                  inform::Ptr{l2rt_inform_type{Float32,
                                                                               Int64}})::Cvoid
end

function l2rt_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.l2rt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{l2rt_control_type{Float64,Int32}},
                                          inform::Ptr{l2rt_inform_type{Float64,Int32}})::Cvoid
end

function l2rt_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.l2rt_terminate_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{l2rt_control_type{Float64,
                                                                               Int64}},
                                                inform::Ptr{l2rt_inform_type{Float64,Int64}})::Cvoid
end

function l2rt_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.l2rt_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{l2rt_control_type{Float128,
                                                                              Int32}},
                                               inform::Ptr{l2rt_inform_type{Float128,Int32}})::Cvoid
end

function l2rt_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.l2rt_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{l2rt_control_type{Float128,
                                                                                    Int64}},
                                                     inform::Ptr{l2rt_inform_type{Float128,
                                                                                  Int64}})::Cvoid
end

function run_sif(::Val{:l2rt}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runl2rt_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:l2rt}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runl2rt_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
