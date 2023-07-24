export convert_control_type

mutable struct convert_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  transpose::Bool
  sum_duplicates::Bool
  order::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}

  convert_control_type() = new()
end

export convert_time_type

mutable struct convert_time_type{T}
  total::T
  clock_total::T

  convert_time_type{T}() where T = new()
end

export convert_inform_type

mutable struct convert_inform_type{T}
  status::Cint
  alloc_status::Cint
  duplicates::Cint
  bad_alloc::NTuple{81,Cchar}
  time::convert_time_type{T}

  convert_inform_type{T}() where T = new()
end

export convert_initialize_s

function convert_initialize_s(data, control, status)
  @ccall libgalahad_single.convert_initialize_s(data::Ptr{Ptr{Cvoid}},
                                                control::Ref{convert_control_type},
                                                status::Ptr{Cint})::Cvoid
end

export convert_initialize

function convert_initialize(data, control, status)
  @ccall libgalahad_double.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ref{convert_control_type},
                                              status::Ptr{Cint})::Cvoid
end

export convert_information_s

function convert_information_s(data, inform, status)
  @ccall libgalahad_single.convert_information_s(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ref{convert_inform_type{Float32}},
                                                 status::Ptr{Cint})::Cvoid
end

export convert_information

function convert_information(data, inform, status)
  @ccall libgalahad_double.convert_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ref{convert_inform_type{Float64}},
                                               status::Ptr{Cint})::Cvoid
end

export convert_terminate_s

function convert_terminate_s(data, control, inform)
  @ccall libgalahad_single.convert_terminate_s(data::Ptr{Ptr{Cvoid}},
                                               control::Ref{convert_control_type},
                                               inform::Ref{convert_inform_type{Float32}})::Cvoid
end

export convert_terminate

function convert_terminate(data, control, inform)
  @ccall libgalahad_double.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{convert_control_type},
                                             inform::Ref{convert_inform_type{Float64}})::Cvoid
end
