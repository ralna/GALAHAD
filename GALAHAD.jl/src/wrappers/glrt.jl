export glrt_control_type

struct glrt_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  itmax::Cint
  stopping_rule::Cint
  freq::Cint
  extra_vectors::Cint
  ritz_printout_device::Cint
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

struct glrt_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  iter_pass2::Cint
  obj::T
  obj_regularized::T
  multiplier::T
  xpo_norm::T
  leftmost::T
  negative_curvature::Bool
  hard_case::Bool
end

export glrt_initialize_s

function glrt_initialize_s(data, control, status)
  @ccall libgalahad_single.glrt_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{glrt_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export glrt_initialize

function glrt_initialize(data, control, status)
  @ccall libgalahad_double.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{glrt_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export glrt_read_specfile_s

function glrt_read_specfile_s(control, specfile)
  @ccall libgalahad_single.glrt_read_specfile_s(control::Ptr{glrt_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export glrt_read_specfile

function glrt_read_specfile(control, specfile)
  @ccall libgalahad_double.glrt_read_specfile(control::Ptr{glrt_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export glrt_import_control_s

function glrt_import_control_s(control, data, status)
  @ccall libgalahad_single.glrt_import_control_s(control::Ptr{glrt_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

export glrt_import_control

function glrt_import_control(control, data, status)
  @ccall libgalahad_double.glrt_import_control(control::Ptr{glrt_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export glrt_solve_problem_s

function glrt_solve_problem_s(data, status, n, power, weight, x, r, vector)
  @ccall libgalahad_single.glrt_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, power::Float32, weight::Float32,
                                                x::Ptr{Float32}, r::Ptr{Float32},
                                                vector::Ptr{Float32})::Cvoid
end

export glrt_solve_problem

function glrt_solve_problem(data, status, n, power, weight, x, r, vector)
  @ccall libgalahad_double.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, power::Float64, weight::Float64,
                                              x::Ptr{Float64}, r::Ptr{Float64},
                                              vector::Ptr{Float64})::Cvoid
end

export glrt_information_s

function glrt_information_s(data, inform, status)
  @ccall libgalahad_single.glrt_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{glrt_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export glrt_information

function glrt_information(data, inform, status)
  @ccall libgalahad_double.glrt_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{glrt_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export glrt_terminate_s

function glrt_terminate_s(data, control, inform)
  @ccall libgalahad_single.glrt_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{glrt_control_type{Float32}},
                                            inform::Ptr{glrt_inform_type{Float32}})::Cvoid
end

export glrt_terminate

function glrt_terminate(data, control, inform)
  @ccall libgalahad_double.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{glrt_control_type{Float64}},
                                          inform::Ptr{glrt_inform_type{Float64}})::Cvoid
end
