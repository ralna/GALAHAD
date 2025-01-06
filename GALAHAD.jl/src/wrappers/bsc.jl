export bsc_control_type

struct bsc_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  max_col::INT
  new_a::INT
  extra_space_s::INT
  s_also_by_column::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export bsc_inform_type

struct bsc_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  max_col_a::INT
  exceeds_max_col::INT
  time::T
  clock_time::T
end

export bsc_initialize

function bsc_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bsc_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function bsc_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function bsc_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bsc_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function bsc_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function bsc_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function bsc_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bsc_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export bsc_information

function bsc_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.bsc_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bsc_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bsc_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.bsc_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bsc_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bsc_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{bsc_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export bsc_terminate

function bsc_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bsc_control_type{Int32}},
                                         inform::Ptr{bsc_inform_type{Float32,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int64}},
                                            inform::Ptr{bsc_inform_type{Float32,Int64}})::Cvoid
end

function bsc_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bsc_control_type{Int32}},
                                         inform::Ptr{bsc_inform_type{Float64,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int64}},
                                            inform::Ptr{bsc_inform_type{Float64,Int64}})::Cvoid
end

function bsc_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int32}},
                                            inform::Ptr{bsc_inform_type{Float128,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{bsc_control_type{Int64}},
                                               inform::Ptr{bsc_inform_type{Float128,Int64}})::Cvoid
end
