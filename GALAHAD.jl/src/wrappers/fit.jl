export fit_control_type

struct fit_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export fit_inform_type

struct fit_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
end

export fit_initialize

function fit_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.fit_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fit_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function fit_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{fit_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export fit_information

function fit_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.fit_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{fit_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function fit_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.fit_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{fit_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export fit_terminate

function fit_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.fit_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{fit_control_type},
                                           inform::Ptr{fit_inform_type})::Cvoid
end

function fit_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{fit_control_type},
                                         inform::Ptr{fit_inform_type})::Cvoid
end
