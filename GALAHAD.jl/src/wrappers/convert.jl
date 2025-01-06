export convert_control_type

struct convert_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
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

struct convert_inform_type{T,INT}
  status::INT
  alloc_status::INT
  duplicates::INT
  bad_alloc::NTuple{81,Cchar}
  time::convert_time_type{T}
end

export convert_initialize

function convert_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{convert_control_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function convert_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{convert_control_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function convert_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{convert_control_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function convert_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{convert_control_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function convert_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{convert_control_type{Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function convert_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{convert_control_type{Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

export convert_information

function convert_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.convert_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{convert_inform_type{Float32,
                                                                               Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function convert_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.convert_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{convert_inform_type{Float32,
                                                                                  Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function convert_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.convert_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{convert_inform_type{Float64,
                                                                               Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function convert_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.convert_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{convert_inform_type{Float64,
                                                                                  Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function convert_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.convert_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{convert_inform_type{Float128,
                                                                                  Int32}},
                                                  status::Ptr{Int32})::Cvoid
end

function convert_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.convert_information(data::Ptr{Ptr{Cvoid}},
                                                     inform::Ptr{convert_inform_type{Float128,
                                                                                     Int64}},
                                                     status::Ptr{Int64})::Cvoid
end

export convert_terminate

function convert_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{convert_control_type{Int32}},
                                             inform::Ptr{convert_inform_type{Float32,Int32}})::Cvoid
end

function convert_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{convert_control_type{Int64}},
                                                inform::Ptr{convert_inform_type{Float32,
                                                                                Int64}})::Cvoid
end

function convert_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{convert_control_type{Int32}},
                                             inform::Ptr{convert_inform_type{Float64,Int32}})::Cvoid
end

function convert_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{convert_control_type{Int64}},
                                                inform::Ptr{convert_inform_type{Float64,
                                                                                Int64}})::Cvoid
end

function convert_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{convert_control_type{Int32}},
                                                inform::Ptr{convert_inform_type{Float128,
                                                                                Int32}})::Cvoid
end

function convert_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{convert_control_type{Int64}},
                                                   inform::Ptr{convert_inform_type{Float128,
                                                                                   Int64}})::Cvoid
end
