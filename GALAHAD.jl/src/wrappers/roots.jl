export roots_control_type

struct roots_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  tol::T
  zero_coef::T
  zero_f::T
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export roots_inform_type

struct roots_inform_type{INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
end

export roots_initialize

function roots_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{roots_control_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function roots_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{roots_control_type{Float32,
                                                                               Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function roots_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{roots_control_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function roots_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{roots_control_type{Float64,
                                                                               Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function roots_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{roots_control_type{Float128,
                                                                               Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function roots_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{roots_control_type{Float128,
                                                                                  Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export roots_information

function roots_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.roots_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{roots_inform_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function roots_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.roots_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{roots_inform_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

function roots_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.roots_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{roots_inform_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function roots_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.roots_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{roots_inform_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

function roots_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.roots_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{roots_inform_type{Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function roots_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.roots_information(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{roots_inform_type{Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

export roots_terminate

function roots_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{roots_control_type{Float32,Int32}},
                                           inform::Ptr{roots_inform_type{Int32}})::Cvoid
end

function roots_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{roots_control_type{Float32,
                                                                              Int64}},
                                              inform::Ptr{roots_inform_type{Int64}})::Cvoid
end

function roots_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{roots_control_type{Float64,Int32}},
                                           inform::Ptr{roots_inform_type{Int32}})::Cvoid
end

function roots_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{roots_control_type{Float64,
                                                                              Int64}},
                                              inform::Ptr{roots_inform_type{Int64}})::Cvoid
end

function roots_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{roots_control_type{Float128,
                                                                              Int32}},
                                              inform::Ptr{roots_inform_type{Int32}})::Cvoid
end

function roots_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{roots_control_type{Float128,
                                                                                 Int64}},
                                                 inform::Ptr{roots_inform_type{Int64}})::Cvoid
end
