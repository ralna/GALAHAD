export scu_control_type

struct scu_control_type
  f_indexing::Bool
end

export scu_inform_type

struct scu_inform_type{INT}
  status::INT
  alloc_status::INT
  inertia::NTuple{3,INT}
end

export scu_initialize

function scu_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{scu_control_type},
                                          status::Ptr{Int32})::Cvoid
end

function scu_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{scu_control_type},
                                             status::Ptr{Int64})::Cvoid
end

function scu_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{scu_control_type},
                                          status::Ptr{Int32})::Cvoid
end

function scu_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{scu_control_type},
                                             status::Ptr{Int64})::Cvoid
end

function scu_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{scu_control_type},
                                             status::Ptr{Int32})::Cvoid
end

function scu_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.scu_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{scu_control_type},
                                                status::Ptr{Int64})::Cvoid
end

export scu_information

function scu_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.scu_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{scu_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function scu_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.scu_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{scu_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function scu_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.scu_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{scu_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function scu_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.scu_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{scu_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function scu_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.scu_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{scu_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function scu_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.scu_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{scu_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export scu_terminate

function scu_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{scu_control_type},
                                         inform::Ptr{scu_inform_type{Int32}})::Cvoid
end

function scu_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{scu_control_type},
                                            inform::Ptr{scu_inform_type{Int64}})::Cvoid
end

function scu_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{scu_control_type},
                                         inform::Ptr{scu_inform_type{Int32}})::Cvoid
end

function scu_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{scu_control_type},
                                            inform::Ptr{scu_inform_type{Int64}})::Cvoid
end

function scu_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{scu_control_type},
                                            inform::Ptr{scu_inform_type{Int32}})::Cvoid
end

function scu_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{scu_control_type},
                                               inform::Ptr{scu_inform_type{Int64}})::Cvoid
end
