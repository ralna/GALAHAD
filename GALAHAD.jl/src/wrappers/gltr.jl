export gltr_control_type

struct gltr_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  itmax::INT
  Lanczos_itmax::INT
  extra_vectors::INT
  ritz_printout_device::INT
  stop_relative::T
  stop_absolute::T
  fraction_opt::T
  f_min::T
  rminvr_zero::T
  f_0::T
  unitm::Bool
  steihaug_toint::Bool
  boundary::Bool
  equality_problem::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  print_ritz_values::Bool
  ritz_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
end

export gltr_inform_type

struct gltr_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  iter_pass2::INT
  obj::T
  multiplier::T
  mnormx::T
  piv::T
  curv::T
  rayleigh::T
  leftmost::T
  negative_curvature::Bool
  hard_case::Bool
end

export gltr_initialize

function gltr_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{gltr_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function gltr_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{gltr_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function gltr_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{gltr_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function gltr_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{gltr_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function gltr_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{gltr_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function gltr_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{gltr_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export gltr_read_specfile

function gltr_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.gltr_read_specfile(control::Ptr{gltr_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function gltr_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.gltr_read_specfile(control::Ptr{gltr_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function gltr_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.gltr_read_specfile(control::Ptr{gltr_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function gltr_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.gltr_read_specfile(control::Ptr{gltr_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function gltr_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.gltr_read_specfile(control::Ptr{gltr_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function gltr_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.gltr_read_specfile(control::Ptr{gltr_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export gltr_import_control

function gltr_import_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.gltr_import_control(control::Ptr{gltr_control_type{Float32,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function gltr_import_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.gltr_import_control(control::Ptr{gltr_control_type{Float32,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function gltr_import_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.gltr_import_control(control::Ptr{gltr_control_type{Float64,
                                                                              Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function gltr_import_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.gltr_import_control(control::Ptr{gltr_control_type{Float64,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64})::Cvoid
end

function gltr_import_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.gltr_import_control(control::Ptr{gltr_control_type{Float128,
                                                                                 Int32}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int32})::Cvoid
end

function gltr_import_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.gltr_import_control(control::Ptr{gltr_control_type{Float128,
                                                                                    Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

export gltr_solve_problem

function gltr_solve_problem(::Type{Float32}, ::Type{Int32}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_single.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, radius::Float32, x::Ptr{Float32},
                                              r::Ptr{Float32}, vector::Ptr{Float32})::Cvoid
end

function gltr_solve_problem(::Type{Float32}, ::Type{Int64}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_single_64.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, radius::Float32, x::Ptr{Float32},
                                                 r::Ptr{Float32},
                                                 vector::Ptr{Float32})::Cvoid
end

function gltr_solve_problem(::Type{Float64}, ::Type{Int32}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_double.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, radius::Float64, x::Ptr{Float64},
                                              r::Ptr{Float64}, vector::Ptr{Float64})::Cvoid
end

function gltr_solve_problem(::Type{Float64}, ::Type{Int64}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_double_64.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, radius::Float64, x::Ptr{Float64},
                                                 r::Ptr{Float64},
                                                 vector::Ptr{Float64})::Cvoid
end

function gltr_solve_problem(::Type{Float128}, ::Type{Int32}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_quadruple.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, radius::Cfloat128,
                                                 x::Ptr{Float128}, r::Ptr{Float128},
                                                 vector::Ptr{Float128})::Cvoid
end

function gltr_solve_problem(::Type{Float128}, ::Type{Int64}, data, status, n, radius, x, r,
                            vector)
  @ccall libgalahad_quadruple_64.gltr_solve_problem(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    radius::Cfloat128, x::Ptr{Float128},
                                                    r::Ptr{Float128},
                                                    vector::Ptr{Float128})::Cvoid
end

export gltr_information

function gltr_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.gltr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{gltr_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function gltr_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.gltr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{gltr_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function gltr_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.gltr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{gltr_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function gltr_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.gltr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{gltr_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function gltr_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.gltr_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{gltr_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function gltr_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.gltr_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{gltr_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export gltr_terminate

function gltr_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gltr_control_type{Float32,Int32}},
                                          inform::Ptr{gltr_inform_type{Float32,Int32}})::Cvoid
end

function gltr_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gltr_control_type{Float32,Int64}},
                                             inform::Ptr{gltr_inform_type{Float32,Int64}})::Cvoid
end

function gltr_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gltr_control_type{Float64,Int32}},
                                          inform::Ptr{gltr_inform_type{Float64,Int32}})::Cvoid
end

function gltr_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gltr_control_type{Float64,Int64}},
                                             inform::Ptr{gltr_inform_type{Float64,Int64}})::Cvoid
end

function gltr_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gltr_control_type{Float128,Int32}},
                                             inform::Ptr{gltr_inform_type{Float128,Int32}})::Cvoid
end

function gltr_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{gltr_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{gltr_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
