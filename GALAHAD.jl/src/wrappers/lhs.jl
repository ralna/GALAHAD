export lhs_control_type

struct lhs_control_type
  error::Cint
  out::Cint
  print_level::Cint
  duplication::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export lhs_inform_type

struct lhs_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
end

export lhs_initialize

function lhs_initialize(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lhs_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lhs_control_type},
                                            inform::Ptr{lhs_inform_type})::Cvoid
end

function lhs_initialize(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lhs_control_type},
                                          inform::Ptr{lhs_inform_type})::Cvoid
end

export lhs_read_specfile

function lhs_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.lhs_read_specfile_s(control::Ptr{lhs_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.lhs_read_specfile(control::Ptr{lhs_control_type},
                                             specfile::Ptr{Cchar})::Cvoid
end

export lhs_ihs

function lhs_ihs(::Type{Float32}, n_dimen, n_points, seed, X, control, inform, data)
  @ccall libgalahad_single.lhs_ihs_s(n_dimen::Cint, n_points::Cint, seed::Ptr{Cint},
                                     X::Ptr{Ptr{Cint}}, control::Ptr{lhs_control_type},
                                     inform::Ptr{lhs_inform_type},
                                     data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float64}, n_dimen, n_points, seed, X, control, inform, data)
  @ccall libgalahad_double.lhs_ihs(n_dimen::Cint, n_points::Cint, seed::Ptr{Cint},
                                   X::Ptr{Ptr{Cint}}, control::Ptr{lhs_control_type},
                                   inform::Ptr{lhs_inform_type},
                                   data::Ptr{Ptr{Cvoid}})::Cvoid
end

export lhs_get_seed

function lhs_get_seed(::Type{Float32}, seed)
  @ccall libgalahad_single.lhs_get_seed_s(seed::Ptr{Cint})::Cvoid
end

function lhs_get_seed(::Type{Float64}, seed)
  @ccall libgalahad_double.lhs_get_seed(seed::Ptr{Cint})::Cvoid
end

export lhs_information

function lhs_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.lhs_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{lhs_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function lhs_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.lhs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lhs_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export lhs_terminate

function lhs_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.lhs_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{lhs_control_type},
                                           inform::Ptr{lhs_inform_type})::Cvoid
end

function lhs_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lhs_control_type},
                                         inform::Ptr{lhs_inform_type})::Cvoid
end
