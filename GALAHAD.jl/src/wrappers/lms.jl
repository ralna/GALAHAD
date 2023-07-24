export lms_control_type

mutable struct lms_control_type{T}
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

  lms_control_type{T}() where T = new()
end

export lms_time_type

mutable struct lms_time_type{T}
  total::T
  setup::T
  form::T
  apply::T
  clock_total::T
  clock_setup::T
  clock_form::T
  clock_apply::T

  lms_time_type{T}() where T = new()
end

export lms_inform_type

mutable struct lms_inform_type{T}
  status::Cint
  alloc_status::Cint
  length::Cint
  updates_skipped::Bool
  bad_alloc::NTuple{81,Cchar}
  time::lms_time_type{T}

  lms_inform_type{T}() where T = new()
end

export lms_initialize_s

function lms_initialize_s(data, control, status)
  @ccall libgalahad_single.lms_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{lms_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export lms_initialize

function lms_initialize(data, control, status)
  @ccall libgalahad_double.lms_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{lms_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export lms_information_s

function lms_information_s(data, inform, status)
  @ccall libgalahad_single.lms_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{lms_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export lms_information

function lms_information(data, inform, status)
  @ccall libgalahad_double.lms_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{lms_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export lms_terminate_s

function lms_terminate_s(data, control, inform)
  @ccall libgalahad_single.lms_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{lms_control_type{Float32}},
                                           inform::Ref{lms_inform_type{Float32}})::Cvoid
end

export lms_terminate

function lms_terminate(data, control, inform)
  @ccall libgalahad_double.lms_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{lms_control_type{Float64}},
                                         inform::Ref{lms_inform_type{Float64}})::Cvoid
end
