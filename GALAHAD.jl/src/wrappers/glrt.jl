export glrt_control_type

struct glrt_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  itmax::INT
  stopping_rule::INT
  freq::INT
  extra_vectors::INT
  ritz_printout_device::INT
  stop_relative::T
  stop_absolute::T
  fraction_opt::T
  rminvr_zero::T
  f_0::T
  unitm::Bool
  impose_descent::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  print_ritz_values::Bool
  ritz_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
end

export glrt_inform_type

struct glrt_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  iter_pass2::INT
  obj::T
  obj_regularized::T
  multiplier::T
  xpo_norm::T
  leftmost::T
  negative_curvature::Bool
  hard_case::Bool
end

export glrt_initialize

function glrt_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{glrt_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function glrt_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{glrt_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function glrt_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{glrt_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function glrt_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{glrt_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function glrt_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{glrt_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function glrt_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{glrt_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export glrt_read_specfile

function glrt_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.glrt_read_specfile(control::Ptr{glrt_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function glrt_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.glrt_read_specfile(control::Ptr{glrt_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function glrt_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.glrt_read_specfile(control::Ptr{glrt_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function glrt_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.glrt_read_specfile(control::Ptr{glrt_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function glrt_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.glrt_read_specfile(control::Ptr{glrt_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function glrt_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.glrt_read_specfile(control::Ptr{glrt_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export glrt_import_control

function glrt_import_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.glrt_import_control(control::Ptr{glrt_control_type{Float32,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function glrt_import_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.glrt_import_control(control::Ptr{glrt_control_type{Float32,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function glrt_import_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.glrt_import_control(control::Ptr{glrt_control_type{Float64,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function glrt_import_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.glrt_import_control(control::Ptr{glrt_control_type{Float64,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function glrt_import_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.glrt_import_control(control::Ptr{glrt_control_type{Float128,
                                                                                 Int32}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int32})::Cvoid
end

function glrt_import_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.glrt_import_control(control::Ptr{glrt_control_type{Float128,
                                                                                    Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

export glrt_solve_problem

function glrt_solve_problem(::Type{Float32}, ::Type{Int32}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_single.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, power::Float32, weight::Float32,
                                              x::Ptr{Float32}, r::Ptr{Float32},
                                              vector::Ptr{Float32})::Cvoid
end

function glrt_solve_problem(::Type{Float32}, ::Type{Int64}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_single_64.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, power::Float32, weight::Float32,
                                                 x::Ptr{Float32}, r::Ptr{Float32},
                                                 vector::Ptr{Float32})::Cvoid
end

function glrt_solve_problem(::Type{Float64}, ::Type{Int32}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_double.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, power::Float64, weight::Float64,
                                              x::Ptr{Float64}, r::Ptr{Float64},
                                              vector::Ptr{Float64})::Cvoid
end

function glrt_solve_problem(::Type{Float64}, ::Type{Int64}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_double_64.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, power::Float64, weight::Float64,
                                                 x::Ptr{Float64}, r::Ptr{Float64},
                                                 vector::Ptr{Float64})::Cvoid
end

function glrt_solve_problem(::Type{Float128}, ::Type{Int32}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_quadruple.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, power::Cfloat128,
                                                 weight::Cfloat128, x::Ptr{Float128},
                                                 r::Ptr{Float128},
                                                 vector::Ptr{Float128})::Cvoid
end

function glrt_solve_problem(::Type{Float128}, ::Type{Int64}, data, status, n, power, weight,
                            x, r, vector)
  @ccall libgalahad_quadruple_64.glrt_solve_problem(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    power::Cfloat128, weight::Cfloat128,
                                                    x::Ptr{Float128}, r::Ptr{Float128},
                                                    vector::Ptr{Float128})::Cvoid
end

export glrt_information

function glrt_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.glrt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{glrt_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function glrt_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.glrt_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{glrt_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function glrt_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.glrt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{glrt_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function glrt_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.glrt_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{glrt_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function glrt_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.glrt_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{glrt_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function glrt_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.glrt_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{glrt_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export glrt_terminate

function glrt_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{glrt_control_type{Float32,Int32}},
                                          inform::Ptr{glrt_inform_type{Float32,Int32}})::Cvoid
end

function glrt_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{glrt_control_type{Float32,Int64}},
                                             inform::Ptr{glrt_inform_type{Float32,Int64}})::Cvoid
end

function glrt_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{glrt_control_type{Float64,Int32}},
                                          inform::Ptr{glrt_inform_type{Float64,Int32}})::Cvoid
end

function glrt_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{glrt_control_type{Float64,Int64}},
                                             inform::Ptr{glrt_inform_type{Float64,Int64}})::Cvoid
end

function glrt_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{glrt_control_type{Float128,Int32}},
                                             inform::Ptr{glrt_inform_type{Float128,Int32}})::Cvoid
end

function glrt_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{glrt_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{glrt_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
