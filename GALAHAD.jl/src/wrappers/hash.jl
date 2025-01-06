export hash_control_type

struct hash_control_type{INT}
  error::INT
  out::INT
  print_level::INT
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export hash_inform_type

struct hash_inform_type{INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
end

export hash_initialize

function hash_initialize(::Type{Float32}, ::Type{Int32}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_single.hash_initialize(nchar::Int32, length::Int32,
                                           data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{hash_control_type{Int32}},
                                           inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_initialize(::Type{Float32}, ::Type{Int64}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_single_64.hash_initialize(nchar::Int64, length::Int64,
                                              data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{hash_control_type{Int64}},
                                              inform::Ptr{hash_inform_type{Int64}})::Cvoid
end

function hash_initialize(::Type{Float64}, ::Type{Int32}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_double.hash_initialize(nchar::Int32, length::Int32,
                                           data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{hash_control_type{Int32}},
                                           inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_initialize(::Type{Float64}, ::Type{Int64}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_double_64.hash_initialize(nchar::Int64, length::Int64,
                                              data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{hash_control_type{Int64}},
                                              inform::Ptr{hash_inform_type{Int64}})::Cvoid
end

function hash_initialize(::Type{Float128}, ::Type{Int32}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_quadruple.hash_initialize(nchar::Int32, length::Int32,
                                              data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{hash_control_type{Int32}},
                                              inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_initialize(::Type{Float128}, ::Type{Int64}, nchar, length, data, control,
                         inform)
  @ccall libgalahad_quadruple_64.hash_initialize(nchar::Int64, length::Int64,
                                                 data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{hash_control_type{Int64}},
                                                 inform::Ptr{hash_inform_type{Int64}})::Cvoid
end

export hash_information

function hash_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.hash_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{hash_inform_type{Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function hash_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.hash_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{hash_inform_type{Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function hash_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.hash_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{hash_inform_type{Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function hash_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.hash_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{hash_inform_type{Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function hash_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.hash_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{hash_inform_type{Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function hash_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.hash_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{hash_inform_type{Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export hash_terminate

function hash_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{hash_control_type{Int32}},
                                          inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{hash_control_type{Int64}},
                                             inform::Ptr{hash_inform_type{Int64}})::Cvoid
end

function hash_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{hash_control_type{Int32}},
                                          inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{hash_control_type{Int64}},
                                             inform::Ptr{hash_inform_type{Int64}})::Cvoid
end

function hash_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{hash_control_type{Int32}},
                                             inform::Ptr{hash_inform_type{Int32}})::Cvoid
end

function hash_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{hash_control_type{Int64}},
                                                inform::Ptr{hash_inform_type{Int64}})::Cvoid
end
