export ir_control_type

mutable struct ir_control_type{T}
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

  ir_control_type{T}() where T = new()
end

export ir_inform_type

mutable struct ir_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  norm_initial_residual::T
  norm_final_residual::T

  ir_inform_type{T}() where T = new()
end

export ir_initialize_s

function ir_initialize_s(data, control, status)
  @ccall libgalahad_single.ir_initialize_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{ir_control_type{Float32}},
                                           status::Ptr{Cint})::Cvoid
end

export ir_initialize

function ir_initialize(data, control, status)
  @ccall libgalahad_double.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{ir_control_type{Float64}},
                                         status::Ptr{Cint})::Cvoid
end

export ir_information_s

function ir_information_s(data, inform, status)
  @ccall libgalahad_single.ir_information_s(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{ir_inform_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export ir_information

function ir_information(data, inform, status)
  @ccall libgalahad_double.ir_information(data::Ptr{Ptr{Cvoid}},
                                          inform::Ref{ir_inform_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export ir_terminate_s

function ir_terminate_s(data, control, inform)
  @ccall libgalahad_single.ir_terminate_s(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{ir_control_type{Float32}},
                                          inform::Ref{ir_inform_type{Float32}})::Cvoid
end

export ir_terminate

function ir_terminate(data, control, inform)
  @ccall libgalahad_double.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                        control::Ref{ir_control_type{Float64}},
                                        inform::Ref{ir_inform_type{Float64}})::Cvoid
end
