export fit_control_type

mutable struct fit_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  fit_control_type() = new()
end

export fit_inform_type

mutable struct fit_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  fit_inform_type() = new()
end

export fit_initialize_s

function fit_initialize_s(data, control, status)
  @ccall libgalahad_single.fit_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{fit_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export fit_initialize

function fit_initialize(data, control, status)
  @ccall libgalahad_double.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{fit_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export fit_information_s

function fit_information_s(data, inform, status)
  @ccall libgalahad_single.fit_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{fit_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export fit_information

function fit_information(data, inform, status)
  @ccall libgalahad_double.fit_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{fit_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export fit_terminate_s

function fit_terminate_s(data, control, inform)
  @ccall libgalahad_single.fit_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{fit_control_type},
                                           inform::Ref{fit_inform_type})::Cvoid
end

export fit_terminate

function fit_terminate(data, control, inform)
  @ccall libgalahad_double.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{fit_control_type},
                                         inform::Ref{fit_inform_type})::Cvoid
end
