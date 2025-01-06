export sec_control_type

struct sec_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  h_initial::T
  update_skip_tol::T
  prefix::NTuple{31,Cchar}
end

export sec_inform_type

struct sec_inform_type{INT}
  status::INT
end

export sec_initialize

function sec_initialize(::Type{Float32}, ::Type{Int32}, control, status)
  @ccall libgalahad_single.sec_initialize(control::Ptr{sec_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sec_initialize(::Type{Float32}, ::Type{Int64}, control, status)
  @ccall libgalahad_single_64.sec_initialize(control::Ptr{sec_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sec_initialize(::Type{Float64}, ::Type{Int32}, control, status)
  @ccall libgalahad_double.sec_initialize(control::Ptr{sec_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sec_initialize(::Type{Float64}, ::Type{Int64}, control, status)
  @ccall libgalahad_double_64.sec_initialize(control::Ptr{sec_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sec_initialize(::Type{Float128}, ::Type{Int32}, control, status)
  @ccall libgalahad_quadruple.sec_initialize(control::Ptr{sec_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function sec_initialize(::Type{Float128}, ::Type{Int64}, control, status)
  @ccall libgalahad_quadruple_64.sec_initialize(control::Ptr{sec_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export sec_information

function sec_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.sec_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sec_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sec_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.sec_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sec_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sec_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.sec_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sec_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sec_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.sec_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sec_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sec_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.sec_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sec_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sec_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.sec_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{sec_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export sec_terminate

function sec_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sec_control_type{Float32,Int32}},
                                         inform::Ptr{sec_inform_type{Int32}})::Cvoid
end

function sec_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sec_control_type{Float32,Int64}},
                                            inform::Ptr{sec_inform_type{Int64}})::Cvoid
end

function sec_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sec_control_type{Float64,Int32}},
                                         inform::Ptr{sec_inform_type{Int32}})::Cvoid
end

function sec_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sec_control_type{Float64,Int64}},
                                            inform::Ptr{sec_inform_type{Int64}})::Cvoid
end

function sec_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sec_control_type{Float128,Int32}},
                                            inform::Ptr{sec_inform_type{Int32}})::Cvoid
end

function sec_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{sec_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{sec_inform_type{Int64}})::Cvoid
end
