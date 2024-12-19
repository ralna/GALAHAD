export sec_control_type

struct sec_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  h_initial::T
  update_skip_tol::T
  prefix::NTuple{31,Cchar}
end

export sec_inform_type

struct sec_inform_type
  status::Cint
end

export sec_initialize

function sec_initialize(::Type{Float32}, control, status)
  @ccall libgalahad_single.sec_initialize_s(control::Ptr{sec_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function sec_initialize(::Type{Float64}, control, status)
  @ccall libgalahad_double.sec_initialize(control::Ptr{sec_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function sec_initialize(::Type{Float128}, control, status)
  @ccall libgalahad_quadruple.sec_initialize_q(control::Ptr{sec_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export sec_information

function sec_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.sec_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{sec_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function sec_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.sec_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sec_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

function sec_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.sec_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{sec_inform_type},
                                                status::Ptr{Cint})::Cvoid
end

export sec_terminate

function sec_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.sec_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sec_control_type{Float32}},
                                           inform::Ptr{sec_inform_type})::Cvoid
end

function sec_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sec_control_type{Float64}},
                                         inform::Ptr{sec_inform_type})::Cvoid
end

function sec_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.sec_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sec_control_type{Float128}},
                                              inform::Ptr{sec_inform_type})::Cvoid
end
