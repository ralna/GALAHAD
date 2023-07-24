export roots_control_type

mutable struct roots_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  tol::T
  zero_coef::T
  zero_f::T
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}

  roots_control_type{T}() where T = new()
end

export roots_inform_type

mutable struct roots_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}

  roots_inform_type() = new()
end

export roots_initialize_s

function roots_initialize_s(data, control, status)
  @ccall libgalahad_single.roots_initialize_s(data::Ptr{Ptr{Cvoid}},
                                              control::Ref{roots_control_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export roots_initialize

function roots_initialize(data, control, status)
  @ccall libgalahad_double.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{roots_control_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export roots_information_s

function roots_information_s(data, inform, status)
  @ccall libgalahad_single.roots_information_s(data::Ptr{Ptr{Cvoid}},
                                               inform::Ref{roots_inform_type},
                                               status::Ptr{Cint})::Cvoid
end

export roots_information

function roots_information(data, inform, status)
  @ccall libgalahad_double.roots_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{roots_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export roots_terminate_s

function roots_terminate_s(data, control, inform)
  @ccall libgalahad_single.roots_terminate_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{roots_control_type{Float32}},
                                             inform::Ref{roots_inform_type})::Cvoid
end

export roots_terminate

function roots_terminate(data, control, inform)
  @ccall libgalahad_double.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{roots_control_type{Float64}},
                                           inform::Ref{roots_inform_type})::Cvoid
end
