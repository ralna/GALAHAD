export fit_control_type

struct fit_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export fit_inform_type

struct fit_inform_type{INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
end

export fit_initialize

function fit_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{fit_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function fit_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fit_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function fit_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{fit_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function fit_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fit_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function fit_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fit_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function fit_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{fit_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export fit_information

function fit_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.fit_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{fit_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function fit_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.fit_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{fit_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function fit_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.fit_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{fit_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function fit_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.fit_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{fit_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function fit_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.fit_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{fit_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function fit_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.fit_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{fit_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export fit_terminate

function fit_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{fit_control_type{Int32}},
                                         inform::Ptr{fit_inform_type{Int32}})::Cvoid
end

function fit_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fit_control_type{Int64}},
                                            inform::Ptr{fit_inform_type{Int64}})::Cvoid
end

function fit_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{fit_control_type{Int32}},
                                         inform::Ptr{fit_inform_type{Int32}})::Cvoid
end

function fit_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fit_control_type{Int64}},
                                            inform::Ptr{fit_inform_type{Int64}})::Cvoid
end

function fit_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fit_control_type{Int32}},
                                            inform::Ptr{fit_inform_type{Int32}})::Cvoid
end

function fit_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{fit_control_type{Int64}},
                                               inform::Ptr{fit_inform_type{Int64}})::Cvoid
end
