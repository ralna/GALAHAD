export sec_control_type

mutable struct sec_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  h_initial::T
  update_skip_tol::T
  prefix::NTuple{31,Cchar}
  sec_control_type{T}() where T = new()
end

export sec_inform_type

mutable struct sec_inform_type
  status::Cint
  sec_inform_type() = new()
end

export sec_initialize_s

function sec_initialize_s(control, status)
  @ccall libgalahad_single.sec_initialize_s(control::Ref{sec_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export sec_initialize

function sec_initialize(control, status)
  @ccall libgalahad_double.sec_initialize(control::Ref{sec_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export sec_information_s

function sec_information_s(data, inform, status)
  @ccall libgalahad_single.sec_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{sec_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export sec_information

function sec_information(data, inform, status)
  @ccall libgalahad_double.sec_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{sec_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export sec_terminate_s

function sec_terminate_s(data, control, inform)
  @ccall libgalahad_single.sec_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sec_control_type{Float32}},
                                           inform::Ref{sec_inform_type})::Cvoid
end

export sec_terminate

function sec_terminate(data, control, inform)
  @ccall libgalahad_double.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{sec_control_type{Float64}},
                                         inform::Ref{sec_inform_type})::Cvoid
end
