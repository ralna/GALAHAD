export fdc_control_type

mutable struct fdc_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  indmin::Cint
  valmin::Cint
  pivot_tol::T
  zero_pivot::T
  max_infeas::T
  use_sls::Bool
  scale::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  unsymmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
  uls_control::uls_control_type{T}
  fdc_control_type{T}() where T = new()
end

export fdc_time_type

mutable struct fdc_time_type{T}
  total::T
  analyse::T
  factorize::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  fdc_time_type{T}() where T = new()
end

export fdc_inform_type

mutable struct fdc_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  non_negligible_pivot::T
  time::fdc_time_type{T}
  sls_inform::sls_inform_type{T}
  uls_inform::uls_inform_type{T}
  fdc_inform_type{T}() where T = new()
end

export fdc_initialize_s

function fdc_initialize_s(data, control, status)
  @ccall libgalahad_single.fdc_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{fdc_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export fdc_initialize

function fdc_initialize(data, control, status)
  @ccall libgalahad_double.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{fdc_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export fdc_read_specfile_s

function fdc_read_specfile_s(control, specfile)
  @ccall libgalahad_single.fdc_read_specfile_s(control::Ref{fdc_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export fdc_read_specfile

function fdc_read_specfile(control, specfile)
  @ccall libgalahad_double.fdc_read_specfile(control::Ref{fdc_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export fdc_find_dependent_rows_s

function fdc_find_dependent_rows_s(control, data, inform, status, m, n, A_ne, A_col, A_ptr,
                                 A_val, b, n_depen, depen)
  @ccall libgalahad_single.fdc_find_dependent_rows_s(control::Ref{fdc_control_type{Float32}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     inform::Ref{fdc_inform_type{Float32}},
                                                     status::Ptr{Cint}, m::Cint, n::Cint,
                                                     A_ne::Cint, A_col::Ptr{Cint},
                                                     A_ptr::Ptr{Cint}, A_val::Ptr{Float32},
                                                     b::Ptr{Float32}, n_depen::Ptr{Cint},
                                                     depen::Ptr{Cint})::Cvoid
end

export fdc_find_dependent_rows

function fdc_find_dependent_rows(control, data, inform, status, m, n, A_ne, A_col, A_ptr,
                               A_val, b, n_depen, depen)
  @ccall libgalahad_double.fdc_find_dependent_rows(control::Ref{fdc_control_type{Float64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   inform::Ref{fdc_inform_type{Float64}},
                                                   status::Ptr{Cint}, m::Cint, n::Cint,
                                                   A_ne::Cint, A_col::Ptr{Cint},
                                                   A_ptr::Ptr{Cint}, A_val::Ptr{Float64},
                                                   b::Ptr{Float64}, n_depen::Ptr{Cint},
                                                   depen::Ptr{Cint})::Cvoid
end

export fdc_terminate_s

function fdc_terminate_s(data, control, inform)
  @ccall libgalahad_single.fdc_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{fdc_control_type{Float32}},
                                           inform::Ref{fdc_inform_type{Float32}})::Cvoid
end

export fdc_terminate

function fdc_terminate(data, control, inform)
  @ccall libgalahad_double.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{fdc_control_type{Float64}},
                                         inform::Ref{fdc_inform_type{Float64}})::Cvoid
end
