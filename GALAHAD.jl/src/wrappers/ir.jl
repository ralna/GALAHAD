export ir_control_type

struct ir_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  itref_max::INT
  acceptable_residual_relative::T
  acceptable_residual_absolute::T
  required_residual_relative::T
  record_residuals::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export ir_inform_type

struct ir_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  norm_initial_residual::T
  norm_final_residual::T
end

export ir_initialize

function ir_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ir_control_type{Float32,Int32}},
                                         status::Ptr{Int32})::Cvoid
end

function ir_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ir_control_type{Float32,Int64}},
                                            status::Ptr{Int64})::Cvoid
end

function ir_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{ir_control_type{Float64,Int32}},
                                         status::Ptr{Int32})::Cvoid
end

function ir_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ir_control_type{Float64,Int64}},
                                            status::Ptr{Int64})::Cvoid
end

function ir_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ir_control_type{Float128,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function ir_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ir_control_type{Float128,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

export ir_information

function ir_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.ir_information(data::Ptr{Ptr{Cvoid}},
                                          inform::Ptr{ir_inform_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function ir_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.ir_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{ir_inform_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function ir_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.ir_information(data::Ptr{Ptr{Cvoid}},
                                          inform::Ptr{ir_inform_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function ir_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.ir_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{ir_inform_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function ir_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.ir_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{ir_inform_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function ir_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.ir_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{ir_inform_type{Float128,Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export ir_terminate

function ir_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{ir_control_type{Float32,Int32}},
                                        inform::Ptr{ir_inform_type{Float32,Int32}})::Cvoid
end

function ir_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ir_control_type{Float32,Int64}},
                                           inform::Ptr{ir_inform_type{Float32,Int64}})::Cvoid
end

function ir_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{ir_control_type{Float64,Int32}},
                                        inform::Ptr{ir_inform_type{Float64,Int32}})::Cvoid
end

function ir_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ir_control_type{Float64,Int64}},
                                           inform::Ptr{ir_inform_type{Float64,Int64}})::Cvoid
end

function ir_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ir_control_type{Float128,Int32}},
                                           inform::Ptr{ir_inform_type{Float128,Int32}})::Cvoid
end

function ir_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{ir_control_type{Float128,Int64}},
                                              inform::Ptr{ir_inform_type{Float128,Int64}})::Cvoid
end
