export bsc_control_type

struct bsc_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  max_col::Cint
  new_a::Cint
  extra_space_s::Cint
  s_also_by_column::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export bsc_inform_type

struct bsc_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  max_col_a::Cint
  exceeds_max_col::Cint
  time::T
  clock_time::T
end

export bsc_initialize

function bsc_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.bsc_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function bsc_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bsc_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export bsc_information

function bsc_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.bsc_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{bsc_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function bsc_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.bsc_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bsc_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export bsc_terminate

function bsc_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.bsc_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bsc_control_type},
                                           inform::Ptr{bsc_inform_type{Float32}})::Cvoid
end

function bsc_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bsc_control_type},
                                         inform::Ptr{bsc_inform_type{Float64}})::Cvoid
end
