export sbls_control_type

struct sbls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  indmin::INT
  valmin::INT
  len_ulsmin::INT
  itref_max::INT
  maxit_pcg::INT
  new_a::INT
  new_h::INT
  new_c::INT
  preconditioner::INT
  semi_bandwidth::INT
  factorization::INT
  max_col::INT
  scaling::INT
  ordering::INT
  pivot_tol::T
  pivot_tol_for_basis::T
  zero_pivot::T
  static_tolerance::T
  static_level::T
  min_diagonal::T
  stop_absolute::T
  stop_relative::T
  remove_dependencies::Bool
  find_basis_by_transpose::Bool
  affine::Bool
  allow_singular::Bool
  perturb_to_make_definite::Bool
  get_norm_residual::Bool
  check_basis::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  definite_linear_solver::NTuple{31,Cchar}
  unsymmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
  uls_control::uls_control_type{T,INT}
end

export sbls_time_type

struct sbls_time_type{T}
  total::T
  form::T
  factorize::T
  apply::T
  clock_total::T
  clock_form::T
  clock_factorize::T
  clock_apply::T
end

export sbls_inform_type

struct sbls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  sort_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  preconditioner::INT
  factorization::INT
  d_plus::INT
  rank::INT
  rank_def::Bool
  perturbed::Bool
  iter_pcg::INT
  norm_residual::T
  alternative::Bool
  time::sbls_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  uls_inform::uls_inform_type{T,INT}
end

export sbls_initialize

function sbls_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sbls_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sbls_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sbls_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sbls_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sbls_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sbls_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sbls_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sbls_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sbls_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sbls_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{sbls_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export sbls_read_specfile

function sbls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.sbls_read_specfile(control::Ptr{sbls_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.sbls_read_specfile(control::Ptr{sbls_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.sbls_read_specfile(control::Ptr{sbls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.sbls_read_specfile(control::Ptr{sbls_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.sbls_read_specfile(control::Ptr{sbls_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.sbls_read_specfile(control::Ptr{sbls_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export sbls_import

function sbls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_single.sbls_import(control::Ptr{sbls_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                       H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                       H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                       A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                       A_ptr::Ptr{Int32}, C_type::Ptr{Cchar}, C_ne::Int32,
                                       C_row::Ptr{Int32}, C_col::Ptr{Int32},
                                       C_ptr::Ptr{Int32})::Cvoid
end

function sbls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_single_64.sbls_import(control::Ptr{sbls_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64}, A_type::Ptr{Cchar},
                                          A_ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                          A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                          C_ne::Int64, C_row::Ptr{Int64}, C_col::Ptr{Int64},
                                          C_ptr::Ptr{Int64})::Cvoid
end

function sbls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_double.sbls_import(control::Ptr{sbls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                       H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                       H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                       A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                       A_ptr::Ptr{Int32}, C_type::Ptr{Cchar}, C_ne::Int32,
                                       C_row::Ptr{Int32}, C_col::Ptr{Int32},
                                       C_ptr::Ptr{Int32})::Cvoid
end

function sbls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_double_64.sbls_import(control::Ptr{sbls_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64}, A_type::Ptr{Cchar},
                                          A_ne::Int64, A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                          A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                          C_ne::Int64, C_row::Ptr{Int64}, C_col::Ptr{Int64},
                                          C_ptr::Ptr{Int64})::Cvoid
end

function sbls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_quadruple.sbls_import(control::Ptr{sbls_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, H_type::Ptr{Cchar},
                                          H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                          H_ptr::Ptr{Int32}, A_type::Ptr{Cchar},
                                          A_ne::Int32, A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                          A_ptr::Ptr{Int32}, C_type::Ptr{Cchar},
                                          C_ne::Int32, C_row::Ptr{Int32}, C_col::Ptr{Int32},
                                          C_ptr::Ptr{Int32})::Cvoid
end

function sbls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_quadruple_64.sbls_import(control::Ptr{sbls_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, H_type::Ptr{Cchar},
                                             H_ne::Int64, H_row::Ptr{Int64},
                                             H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                             A_type::Ptr{Cchar}, A_ne::Int64,
                                             A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                             A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                             C_ne::Int64, C_row::Ptr{Int64},
                                             C_col::Ptr{Int64}, C_ptr::Ptr{Int64})::Cvoid
end

export sbls_reset_control

function sbls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.sbls_reset_control(control::Ptr{sbls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function sbls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.sbls_reset_control(control::Ptr{sbls_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function sbls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.sbls_reset_control(control::Ptr{sbls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function sbls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.sbls_reset_control(control::Ptr{sbls_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function sbls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.sbls_reset_control(control::Ptr{sbls_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function sbls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.sbls_reset_control(control::Ptr{sbls_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export sbls_factorize_matrix

function sbls_factorize_matrix(::Type{Float32}, ::Type{Int32}, data, status, n, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_single.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, h_ne::Int32, H_val::Ptr{Float32},
                                                 a_ne::Int32, A_val::Ptr{Float32},
                                                 c_ne::Int32, C_val::Ptr{Float32},
                                                 D::Ptr{Float32})::Cvoid
end

function sbls_factorize_matrix(::Type{Float32}, ::Type{Int64}, data, status, n, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_single_64.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    h_ne::Int64, H_val::Ptr{Float32},
                                                    a_ne::Int64, A_val::Ptr{Float32},
                                                    c_ne::Int64, C_val::Ptr{Float32},
                                                    D::Ptr{Float32})::Cvoid
end

function sbls_factorize_matrix(::Type{Float64}, ::Type{Int32}, data, status, n, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_double.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, h_ne::Int32, H_val::Ptr{Float64},
                                                 a_ne::Int32, A_val::Ptr{Float64},
                                                 c_ne::Int32, C_val::Ptr{Float64},
                                                 D::Ptr{Float64})::Cvoid
end

function sbls_factorize_matrix(::Type{Float64}, ::Type{Int64}, data, status, n, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_double_64.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    h_ne::Int64, H_val::Ptr{Float64},
                                                    a_ne::Int64, A_val::Ptr{Float64},
                                                    c_ne::Int64, C_val::Ptr{Float64},
                                                    D::Ptr{Float64})::Cvoid
end

function sbls_factorize_matrix(::Type{Float128}, ::Type{Int32}, data, status, n, h_ne,
                               H_val, a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_quadruple.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    h_ne::Int32, H_val::Ptr{Float128},
                                                    a_ne::Int32, A_val::Ptr{Float128},
                                                    c_ne::Int32, C_val::Ptr{Float128},
                                                    D::Ptr{Float128})::Cvoid
end

function sbls_factorize_matrix(::Type{Float128}, ::Type{Int64}, data, status, n, h_ne,
                               H_val, a_ne, A_val, c_ne, C_val, D)
  @ccall libgalahad_quadruple_64.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       h_ne::Int64, H_val::Ptr{Float128},
                                                       a_ne::Int64, A_val::Ptr{Float128},
                                                       c_ne::Int64, C_val::Ptr{Float128},
                                                       D::Ptr{Float128})::Cvoid
end

export sbls_solve_system

function sbls_solve_system(::Type{Float32}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_single.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, m::Int32, sol::Ptr{Float32})::Cvoid
end

function sbls_solve_system(::Type{Float32}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_single_64.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, m::Int64,
                                                sol::Ptr{Float32})::Cvoid
end

function sbls_solve_system(::Type{Float64}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_double.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, m::Int32, sol::Ptr{Float64})::Cvoid
end

function sbls_solve_system(::Type{Float64}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_double_64.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, m::Int64,
                                                sol::Ptr{Float64})::Cvoid
end

function sbls_solve_system(::Type{Float128}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_quadruple.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, m::Int32,
                                                sol::Ptr{Float128})::Cvoid
end

function sbls_solve_system(::Type{Float128}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_quadruple_64.sbls_solve_system(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, m::Int64,
                                                   sol::Ptr{Float128})::Cvoid
end

export sbls_information

function sbls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.sbls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{sbls_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function sbls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.sbls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{sbls_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function sbls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.sbls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{sbls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function sbls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.sbls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{sbls_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function sbls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.sbls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{sbls_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function sbls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.sbls_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{sbls_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export sbls_terminate

function sbls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sbls_control_type{Float32,Int32}},
                                          inform::Ptr{sbls_inform_type{Float32,Int32}})::Cvoid
end

function sbls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sbls_control_type{Float32,Int64}},
                                             inform::Ptr{sbls_inform_type{Float32,Int64}})::Cvoid
end

function sbls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sbls_control_type{Float64,Int32}},
                                          inform::Ptr{sbls_inform_type{Float64,Int32}})::Cvoid
end

function sbls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sbls_control_type{Float64,Int64}},
                                             inform::Ptr{sbls_inform_type{Float64,Int64}})::Cvoid
end

function sbls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sbls_control_type{Float128,Int32}},
                                             inform::Ptr{sbls_inform_type{Float128,Int32}})::Cvoid
end

function sbls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sbls_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{sbls_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
