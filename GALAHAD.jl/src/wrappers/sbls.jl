export sbls_control_type

struct sbls_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  indmin::Cint
  valmin::Cint
  len_ulsmin::Cint
  itref_max::Cint
  maxit_pcg::Cint
  new_a::Cint
  new_h::Cint
  new_c::Cint
  preconditioner::Cint
  semi_bandwidth::Cint
  factorization::Cint
  max_col::Cint
  scaling::Cint
  ordering::Cint
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
  sls_control::sls_control_type{T}
  uls_control::uls_control_type{T}
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

struct sbls_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  sort_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  preconditioner::Cint
  factorization::Cint
  d_plus::Cint
  rank::Cint
  rank_def::Bool
  perturbed::Bool
  iter_pcg::Cint
  norm_residual::T
  alternative::Bool
  time::sbls_time_type{T}
  sls_inform::sls_inform_type{T}
  uls_inform::uls_inform_type{T}
end

export sbls_initialize

function sbls_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.sbls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sbls_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function sbls_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.sbls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sbls_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function sbls_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.sbls_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sbls_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export sbls_read_specfile

function sbls_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.sbls_read_specfile_s(control::Ptr{sbls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.sbls_read_specfile(control::Ptr{sbls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function sbls_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.sbls_read_specfile_q(control::Ptr{sbls_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export sbls_import

function sbls_import(::Type{Float32}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row,
                     C_col, C_ptr)
  @ccall libgalahad_single.sbls_import_s(control::Ptr{sbls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                         H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                         H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint}, C_type::Ptr{Cchar}, C_ne::Cint,
                                         C_row::Ptr{Cint}, C_col::Ptr{Cint},
                                         C_ptr::Ptr{Cint})::Cvoid
end

function sbls_import(::Type{Float64}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row,
                     C_col, C_ptr)
  @ccall libgalahad_double.sbls_import(control::Ptr{sbls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                       H_row::Ptr{Cint}, H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                       A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                       A_col::Ptr{Cint}, A_ptr::Ptr{Cint},
                                       C_type::Ptr{Cchar}, C_ne::Cint, C_row::Ptr{Cint},
                                       C_col::Ptr{Cint}, C_ptr::Ptr{Cint})::Cvoid
end

function sbls_import(::Type{Float128}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type, C_ne, C_row,
                     C_col, C_ptr)
  @ccall libgalahad_quadruple.sbls_import_q(control::Ptr{sbls_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, H_type::Ptr{Cchar},
                                            H_ne::Cint, H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                            H_ptr::Ptr{Cint}, A_type::Ptr{Cchar},
                                            A_ne::Cint, A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                            A_ptr::Ptr{Cint}, C_type::Ptr{Cchar},
                                            C_ne::Cint, C_row::Ptr{Cint}, C_col::Ptr{Cint},
                                            C_ptr::Ptr{Cint})::Cvoid
end

export sbls_reset_control

function sbls_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.sbls_reset_control_s(control::Ptr{sbls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function sbls_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.sbls_reset_control(control::Ptr{sbls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function sbls_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.sbls_reset_control_q(control::Ptr{sbls_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export sbls_factorize_matrix

function sbls_factorize_matrix(::Type{Float32}, data, status, n, h_ne, H_val, a_ne, A_val,
                               c_ne, C_val, D)
  @ccall libgalahad_single.sbls_factorize_matrix_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                                   a_ne::Cint, A_val::Ptr{Float32},
                                                   c_ne::Cint, C_val::Ptr{Float32},
                                                   D::Ptr{Float32})::Cvoid
end

function sbls_factorize_matrix(::Type{Float64}, data, status, n, h_ne, H_val, a_ne, A_val,
                               c_ne, C_val, D)
  @ccall libgalahad_double.sbls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                                 a_ne::Cint, A_val::Ptr{Float64},
                                                 c_ne::Cint, C_val::Ptr{Float64},
                                                 D::Ptr{Float64})::Cvoid
end

function sbls_factorize_matrix(::Type{Float128}, data, status, n, h_ne, H_val, a_ne, A_val,
                               c_ne, C_val, D)
  @ccall libgalahad_quadruple.sbls_factorize_matrix_q(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      h_ne::Cint, H_val::Ptr{Float128},
                                                      a_ne::Cint, A_val::Ptr{Float128},
                                                      c_ne::Cint, C_val::Ptr{Float128},
                                                      D::Ptr{Float128})::Cvoid
end

export sbls_solve_system

function sbls_solve_system(::Type{Float32}, data, status, n, m, sol)
  @ccall libgalahad_single.sbls_solve_system_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, m::Cint, sol::Ptr{Float32})::Cvoid
end

function sbls_solve_system(::Type{Float64}, data, status, n, m, sol)
  @ccall libgalahad_double.sbls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, sol::Ptr{Float64})::Cvoid
end

function sbls_solve_system(::Type{Float128}, data, status, n, m, sol)
  @ccall libgalahad_quadruple.sbls_solve_system_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, m::Cint,
                                                  sol::Ptr{Float128})::Cvoid
end

export sbls_information

function sbls_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.sbls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sbls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function sbls_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.sbls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{sbls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function sbls_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.sbls_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{sbls_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export sbls_terminate

function sbls_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.sbls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sbls_control_type{Float32}},
                                            inform::Ptr{sbls_inform_type{Float32}})::Cvoid
end

function sbls_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.sbls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sbls_control_type{Float64}},
                                          inform::Ptr{sbls_inform_type{Float64}})::Cvoid
end

function sbls_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.sbls_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{sbls_control_type{Float128}},
                                               inform::Ptr{sbls_inform_type{Float128}})::Cvoid
end
