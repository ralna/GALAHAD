export lhs_control_type

struct lhs_control_type{INT}
  error::INT
  out::INT
  print_level::INT
  duplication::INT
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export lhs_inform_type

struct lhs_inform_type{INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
end

export lhs_initialize

function lhs_initialize(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lhs_control_type{Int32}},
                                          inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_initialize(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lhs_control_type{Int64}},
                                             inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end

function lhs_initialize(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{lhs_control_type{Int32}},
                                          inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_initialize(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lhs_control_type{Int64}},
                                             inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end

function lhs_initialize(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lhs_control_type{Int32}},
                                             inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_initialize(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.lhs_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{lhs_control_type{Int64}},
                                                inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end

export lhs_read_specfile

function lhs_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.lhs_read_specfile(control::Ptr{lhs_control_type{Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.lhs_read_specfile(control::Ptr{lhs_control_type{Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.lhs_read_specfile(control::Ptr{lhs_control_type{Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.lhs_read_specfile(control::Ptr{lhs_control_type{Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.lhs_read_specfile(control::Ptr{lhs_control_type{Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lhs_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.lhs_read_specfile(control::Ptr{lhs_control_type{Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export lhs_ihs

function lhs_ihs(::Type{Float32}, ::Type{Int32}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_single.lhs_ihs(n_dimen::Int32, n_points::Int32, seed::Ptr{Int32},
                                   X::Ptr{Int32}, control::Ptr{lhs_control_type{Int32}},
                                   inform::Ptr{lhs_inform_type{Int32}},
                                   data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float32}, ::Type{Int64}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_single_64.lhs_ihs(n_dimen::Int64, n_points::Int64, seed::Ptr{Int64},
                                      X::Ptr{Int64}, control::Ptr{lhs_control_type{Int64}},
                                      inform::Ptr{lhs_inform_type{Int64}},
                                      data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float64}, ::Type{Int32}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_double.lhs_ihs(n_dimen::Int32, n_points::Int32, seed::Ptr{Int32},
                                   X::Ptr{Int32}, control::Ptr{lhs_control_type{Int32}},
                                   inform::Ptr{lhs_inform_type{Int32}},
                                   data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float64}, ::Type{Int64}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_double_64.lhs_ihs(n_dimen::Int64, n_points::Int64, seed::Ptr{Int64},
                                      X::Ptr{Int64}, control::Ptr{lhs_control_type{Int64}},
                                      inform::Ptr{lhs_inform_type{Int64}},
                                      data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float128}, ::Type{Int32}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_quadruple.lhs_ihs(n_dimen::Int32, n_points::Int32, seed::Ptr{Int32},
                                      X::Ptr{Int32}, control::Ptr{lhs_control_type{Int32}},
                                      inform::Ptr{lhs_inform_type{Int32}},
                                      data::Ptr{Ptr{Cvoid}})::Cvoid
end

function lhs_ihs(::Type{Float128}, ::Type{Int64}, n_dimen, n_points, seed, X, control,
                 inform, data)
  @ccall libgalahad_quadruple_64.lhs_ihs(n_dimen::Int64, n_points::Int64, seed::Ptr{Int64},
                                         X::Ptr{Int64},
                                         control::Ptr{lhs_control_type{Int64}},
                                         inform::Ptr{lhs_inform_type{Int64}},
                                         data::Ptr{Ptr{Cvoid}})::Cvoid
end

export lhs_get_seed

function lhs_get_seed(::Type{Float32}, ::Type{Int32}, seed)
  @ccall libgalahad_single.lhs_get_seed(seed::Ptr{Int32})::Cvoid
end

function lhs_get_seed(::Type{Float32}, ::Type{Int64}, seed)
  @ccall libgalahad_single_64.lhs_get_seed(seed::Ptr{Int64})::Cvoid
end

function lhs_get_seed(::Type{Float64}, ::Type{Int32}, seed)
  @ccall libgalahad_double.lhs_get_seed(seed::Ptr{Int32})::Cvoid
end

function lhs_get_seed(::Type{Float64}, ::Type{Int64}, seed)
  @ccall libgalahad_double_64.lhs_get_seed(seed::Ptr{Int64})::Cvoid
end

function lhs_get_seed(::Type{Float128}, ::Type{Int32}, seed)
  @ccall libgalahad_quadruple.lhs_get_seed(seed::Ptr{Int32})::Cvoid
end

function lhs_get_seed(::Type{Float128}, ::Type{Int64}, seed)
  @ccall libgalahad_quadruple_64.lhs_get_seed(seed::Ptr{Int64})::Cvoid
end

export lhs_information

function lhs_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.lhs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lhs_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lhs_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.lhs_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lhs_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lhs_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.lhs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{lhs_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function lhs_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.lhs_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lhs_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function lhs_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.lhs_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lhs_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function lhs_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.lhs_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{lhs_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export lhs_terminate

function lhs_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lhs_control_type{Int32}},
                                         inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lhs_control_type{Int64}},
                                            inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end

function lhs_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{lhs_control_type{Int32}},
                                         inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lhs_control_type{Int64}},
                                            inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end

function lhs_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lhs_control_type{Int32}},
                                            inform::Ptr{lhs_inform_type{Int32}})::Cvoid
end

function lhs_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.lhs_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{lhs_control_type{Int64}},
                                               inform::Ptr{lhs_inform_type{Int64}})::Cvoid
end
