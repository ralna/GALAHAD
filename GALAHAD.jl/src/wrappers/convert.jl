export convert_control_type

struct convert_control_type
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
end

export convert_time_type

struct convert_time_type{T}
  total::T
  clock_total::T
end

export convert_inform_type

struct convert_inform_type{T}
  status::Cint
  alloc_status::Cint
  duplicates::Cint
  bad_alloc::NTuple{81,Cchar}
  time::convert_time_type{T}
end

export convert_initialize

function convert_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.convert_initialize_s(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{convert_control_type},
                                                status::Ptr{Cint})::Cvoid
end

function convert_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{convert_control_type},
                                              status::Ptr{Cint})::Cvoid
end

function convert_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.convert_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{convert_control_type},
                                                   status::Ptr{Cint})::Cvoid
end

export convert_information

function convert_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.convert_information_s(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{convert_inform_type{Float32}},
                                                 status::Ptr{Cint})::Cvoid
end

function convert_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.convert_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{convert_inform_type{Float64}},
                                               status::Ptr{Cint})::Cvoid
end

function convert_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.convert_information_q(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{convert_inform_type{Float128}},
                                                    status::Ptr{Cint})::Cvoid
end

export convert_terminate

function convert_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.convert_terminate_s(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{convert_control_type},
                                               inform::Ptr{convert_inform_type{Float32}})::Cvoid
end

function convert_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{convert_control_type},
                                             inform::Ptr{convert_inform_type{Float64}})::Cvoid
end

function convert_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.convert_terminate_q(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{convert_control_type},
                                                  inform::Ptr{convert_inform_type{Float128}})::Cvoid
end
