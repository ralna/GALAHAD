export roots_control_type

struct roots_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  tol::T
  zero_coef::T
  zero_f::T
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export roots_inform_type

struct roots_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
end

export roots_initialize

function roots_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.roots_initialize_s(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{roots_control_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function roots_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{roots_control_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export roots_information

function roots_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.roots_information_s(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{roots_inform_type},
                                               status::Ptr{Cint})::Cvoid
end

function roots_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.roots_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{roots_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export roots_terminate

function roots_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.roots_terminate_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{roots_control_type{Float32}},
                                             inform::Ptr{roots_inform_type})::Cvoid
end

function roots_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{roots_control_type{Float64}},
                                           inform::Ptr{roots_inform_type})::Cvoid
end
