export lms_control_type

struct lms_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  memory_length::Cint
  method::Cint
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

struct lms_inform_type{T}
  status::Cint
  alloc_status::Cint
  length::Cint
  updates_skipped::Bool
  bad_alloc::NTuple{81,Cchar}
  time::lms_time_type{T}
end

export lms_initialize

function lms_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.lms_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lms_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function lms_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lms_control_type},
                                          status::Ptr{Cint})::Cvoid
end

function lms_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.lms_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lms_control_type},
                                               status::Ptr{Cint})::Cvoid
end

export lms_information

function lms_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.lms_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{lms_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function lms_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.lms_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lms_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function lms_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.lms_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{lms_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export lms_terminate

function lms_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lms_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lms_control_type},
                                           inform::Ptr{lms_inform_type{Float32}})::Cvoid
end

function lms_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lms_control_type},
                                         inform::Ptr{lms_inform_type{Float64}})::Cvoid
end

function lms_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.lms_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{lms_control_type},
                                              inform::Ptr{lms_inform_type{Float128}})::Cvoid
end
