export lms_control_type

struct lms_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  memory_length::INT
  method::INT
  any_method::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export lms_time_type

struct lms_time_type{T}
  total::T
  setup::T
  form::T
  apply::T
  clock_total::T
  clock_setup::T
  clock_form::T
  clock_apply::T
end

export lms_inform_type

struct lms_inform_type{T,INT}
  status::INT
  alloc_status::INT
  length::INT
  updates_skipped::Bool
  bad_alloc::NTuple{81,Cchar}
  time::lms_time_type{T}
end

export lms_initialize

function lms_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lms_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function lms_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lms_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function lms_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lms_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function lms_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lms_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function lms_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lms_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function lms_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{lms_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export lms_information

function lms_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.lms_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lms_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lms_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.lms_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lms_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lms_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.lms_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lms_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lms_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.lms_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lms_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lms_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.lms_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lms_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function lms_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.lms_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{lms_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export lms_terminate

function lms_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lms_control_type{Int32}},
                                         inform::Ptr{lms_inform_type{Float32,Int32}})::Cvoid
end

function lms_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lms_control_type{Int64}},
                                            inform::Ptr{lms_inform_type{Float32,Int64}})::Cvoid
end

function lms_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lms_control_type{Int32}},
                                         inform::Ptr{lms_inform_type{Float64,Int32}})::Cvoid
end

function lms_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lms_control_type{Int64}},
                                            inform::Ptr{lms_inform_type{Float64,Int64}})::Cvoid
end

function lms_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lms_control_type{Int32}},
                                            inform::Ptr{lms_inform_type{Float128,Int32}})::Cvoid
end

function lms_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lms_control_type{Int64}},
                                               inform::Ptr{lms_inform_type{Float128,Int64}})::Cvoid
end
