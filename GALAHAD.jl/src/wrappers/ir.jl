export ir_control_type

struct ir_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  itref_max::Cint
  acceptable_residual_relative::T
  acceptable_residual_absolute::T
  required_residual_relative::T
  record_residuals::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export ir_inform_type

struct ir_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  norm_initial_residual::T
  norm_final_residual::T
end

export ir_initialize

function ir_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.ir_initialize_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ir_control_type{Float32}},
                                           status::Ptr{Cint})::Cvoid
end

function ir_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ir_control_type{Float64}},
                                         status::Ptr{Cint})::Cvoid
end

function ir_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.ir_initialize_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{ir_control_type{Float128}},
                                              status::Ptr{Cint})::Cvoid
end

export ir_information

function ir_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.ir_information_s(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{ir_inform_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function ir_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.ir_information(data::Ptr{Ptr{Cvoid}},
                                          inform::Ptr{ir_inform_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function ir_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.ir_information_q(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{ir_inform_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export ir_terminate

function ir_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.ir_terminate_s(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ir_control_type{Float32}},
                                          inform::Ptr{ir_inform_type{Float32}})::Cvoid
end

function ir_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{ir_control_type{Float64}},
                                        inform::Ptr{ir_inform_type{Float64}})::Cvoid
end

function ir_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.ir_terminate_q(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ir_control_type{Float128}},
                                             inform::Ptr{ir_inform_type{Float128}})::Cvoid
end
