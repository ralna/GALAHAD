export scu_control_type

mutable struct scu_control_type
  f_indexing::Bool
  scu_control_type() = new()
end

export scu_inform_type

mutable struct scu_inform_type
  status::Cint
  alloc_status::Cint
  inertia::NTuple{3,Cint}
  scu_inform_type() = new()
end

export scu_information_s

function scu_information_s(data, inform, status)
  @ccall libgalahad_single.scu_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{scu_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export scu_information

function scu_information(data, inform, status)
  @ccall libgalahad_double.scu_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{scu_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export scu_terminate_s

function scu_terminate_s(data, control, inform)
  @ccall libgalahad_single.scu_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{scu_control_type},
                                           inform::Ref{scu_inform_type})::Cvoid
end

export scu_terminate

function scu_terminate(data, control, inform)
  @ccall libgalahad_double.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{scu_control_type},
                                         inform::Ref{scu_inform_type})::Cvoid
end
