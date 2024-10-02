export scu_control_type

struct scu_control_type
  f_indexing::Bool
end

export scu_inform_type

struct scu_inform_type
  status::Cint
  alloc_status::Cint
  inertia::NTuple{3,Cint}
end

export scu_initialize

function scu_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.scu_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{scu_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function scu_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{scu_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export scu_information

function scu_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.scu_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{scu_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function scu_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.scu_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{scu_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export scu_terminate

function scu_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.scu_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{scu_control_type},
                                           inform::Ptr{scu_inform_type})::Cvoid
end

function scu_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{scu_control_type},
                                         inform::Ptr{scu_inform_type})::Cvoid
end
