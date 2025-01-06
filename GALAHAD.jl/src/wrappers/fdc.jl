export fdc_control_type

struct fdc_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  indmin::INT
  valmin::INT
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
  sls_control::sls_control_type{T,INT}
  uls_control::uls_control_type{T,INT}
end

export fdc_time_type

struct fdc_time_type{T}
  total::T
  analyse::T
  factorize::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
end

export fdc_inform_type

struct fdc_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  factorization_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  non_negligible_pivot::T
  time::fdc_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  uls_inform::uls_inform_type{T,INT}
end

export fdc_initialize

function fdc_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{fdc_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function fdc_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fdc_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function fdc_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{fdc_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function fdc_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fdc_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function fdc_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{fdc_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function fdc_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.fdc_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{fdc_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export fdc_read_specfile

function fdc_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.fdc_read_specfile(control::Ptr{fdc_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function fdc_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.fdc_read_specfile(control::Ptr{fdc_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function fdc_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.fdc_read_specfile(control::Ptr{fdc_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function fdc_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.fdc_read_specfile(control::Ptr{fdc_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function fdc_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.fdc_read_specfile(control::Ptr{fdc_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function fdc_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.fdc_read_specfile(control::Ptr{fdc_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export fdc_find_dependent_rows

function fdc_find_dependent_rows(::Type{Float32}, ::Type{Int32}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_single.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float32,
                                                                                 Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{fdc_inform_type{Float32,
                                                                               Int32}},
                                                   status::Ptr{Int32}, m::Int32, n::Int32,
                                                   A_ne::Int32, A_col::Ptr{Int32},
                                                   A_ptr::Ptr{Int32}, A_val::Ptr{Float32},
                                                   b::Ptr{Float32}, n_depen::Ptr{Int32},
                                                   depen::Ptr{Int32})::Cvoid
end

function fdc_find_dependent_rows(::Type{Float32}, ::Type{Int64}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_single_64.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float32,
                                                                                    Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{fdc_inform_type{Float32,
                                                                                  Int64}},
                                                      status::Ptr{Int64}, m::Int64,
                                                      n::Int64, A_ne::Int64,
                                                      A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                      A_val::Ptr{Float32}, b::Ptr{Float32},
                                                      n_depen::Ptr{Int64},
                                                      depen::Ptr{Int64})::Cvoid
end

function fdc_find_dependent_rows(::Type{Float64}, ::Type{Int32}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_double.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float64,
                                                                                 Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{fdc_inform_type{Float64,
                                                                               Int32}},
                                                   status::Ptr{Int32}, m::Int32, n::Int32,
                                                   A_ne::Int32, A_col::Ptr{Int32},
                                                   A_ptr::Ptr{Int32}, A_val::Ptr{Float64},
                                                   b::Ptr{Float64}, n_depen::Ptr{Int32},
                                                   depen::Ptr{Int32})::Cvoid
end

function fdc_find_dependent_rows(::Type{Float64}, ::Type{Int64}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_double_64.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float64,
                                                                                    Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{fdc_inform_type{Float64,
                                                                                  Int64}},
                                                      status::Ptr{Int64}, m::Int64,
                                                      n::Int64, A_ne::Int64,
                                                      A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                      A_val::Ptr{Float64}, b::Ptr{Float64},
                                                      n_depen::Ptr{Int64},
                                                      depen::Ptr{Int64})::Cvoid
end

function fdc_find_dependent_rows(::Type{Float128}, ::Type{Int32}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_quadruple.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float128,
                                                                                    Int32}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{fdc_inform_type{Float128,
                                                                                  Int32}},
                                                      status::Ptr{Int32}, m::Int32,
                                                      n::Int32, A_ne::Int32,
                                                      A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                      A_val::Ptr{Float128},
                                                      b::Ptr{Float128}, n_depen::Ptr{Int32},
                                                      depen::Ptr{Int32})::Cvoid
end

function fdc_find_dependent_rows(::Type{Float128}, ::Type{Int64}, control, data, inform,
                                 status, m, n, A_ne, A_col, A_ptr, A_val, b, n_depen, depen)
  @ccall libgalahad_quadruple_64.fdc_find_dependent_rows(control::Ptr{fdc_control_type{Float128,
                                                                                       Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         inform::Ptr{fdc_inform_type{Float128,
                                                                                     Int64}},
                                                         status::Ptr{Int64}, m::Int64,
                                                         n::Int64, A_ne::Int64,
                                                         A_col::Ptr{Int64},
                                                         A_ptr::Ptr{Int64},
                                                         A_val::Ptr{Float128},
                                                         b::Ptr{Float128},
                                                         n_depen::Ptr{Int64},
                                                         depen::Ptr{Int64})::Cvoid
end

export fdc_terminate

function fdc_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{fdc_control_type{Float32,Int32}},
                                         inform::Ptr{fdc_inform_type{Float32,Int32}})::Cvoid
end

function fdc_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fdc_control_type{Float32,Int64}},
                                            inform::Ptr{fdc_inform_type{Float32,Int64}})::Cvoid
end

function fdc_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{fdc_control_type{Float64,Int32}},
                                         inform::Ptr{fdc_inform_type{Float64,Int32}})::Cvoid
end

function fdc_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fdc_control_type{Float64,Int64}},
                                            inform::Ptr{fdc_inform_type{Float64,Int64}})::Cvoid
end

function fdc_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fdc_control_type{Float128,Int32}},
                                            inform::Ptr{fdc_inform_type{Float128,Int32}})::Cvoid
end

function fdc_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.fdc_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{fdc_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{fdc_inform_type{Float128,Int64}})::Cvoid
end
