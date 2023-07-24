export tru_control_type

mutable struct tru_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  alive_unit::Cint
  alive_file::NTuple{31,Cchar}
  non_monotone::Cint
  model::Cint
  norm::Cint
  semi_bandwidth::Cint
  lbfgs_vectors::Cint
  max_dxg::Cint
  icfs_vectors::Cint
  mi28_lsize::Cint
  mi28_rsize::Cint
  stop_g_absolute::T
  stop_g_relative::T
  stop_s::T
  advanced_start::Cint
  initial_radius::T
  maximum_radius::T
  eta_successful::T
  eta_very_successful::T
  eta_too_successful::T
  radius_increase::T
  radius_reduce::T
  radius_reduce_max::T
  obj_unbounded::T
  cpu_time_limit::T
  clock_time_limit::T
  hessian_available::Bool
  subproblem_direct::Bool
  retrospective_trust_region::Bool
  renormalize_radius::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  trs_control::trs_control_type{T}
  gltr_control::gltr_control_type{T}
  dps_control::dps_control_type{T}
  psls_control::psls_control_type{T}
  lms_control::lms_control_type{T}
  lms_control_prec::lms_control_type{T}
  sec_control::sec_control_type{T}
  sha_control::sha_control_type

  tru_control_type{T}() where T = new()
end

export tru_time_type

mutable struct tru_time_type{T}
  total::Float32
  preprocess::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
  clock_total::T
  clock_preprocess::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T

  tru_time_type{T}() where T = new()
end

export tru_inform_type

mutable struct tru_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  cg_iter::Cint
  f_eval::Cint
  g_eval::Cint
  h_eval::Cint
  factorization_max::Cint
  factorization_status::Cint
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_g::T
  radius::T
  time::tru_time_type{T}
  trs_inform::trs_inform_type{T}
  gltr_inform::gltr_inform_type{T}
  dps_inform::dps_inform_type{T}
  psls_inform::psls_inform_type{T}
  lms_inform::lms_inform_type{T}
  lms_inform_prec::lms_inform_type{T}
  sec_inform::sec_inform_type
  sha_inform::sha_inform_type

  tru_inform_type{T}() where T = new()
end

export tru_initialize_s

function tru_initialize_s(data, control, status)
  @ccall libgalahad_single.tru_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{tru_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export tru_initialize

function tru_initialize(data, control, status)
  @ccall libgalahad_double.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{tru_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export tru_read_specfile_s

function tru_read_specfile_s(control, specfile)
  @ccall libgalahad_single.tru_read_specfile_s(control::Ref{tru_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export tru_read_specfile

function tru_read_specfile(control, specfile)
  @ccall libgalahad_double.tru_read_specfile(control::Ref{tru_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export tru_import_s

function tru_import_s(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.tru_import_s(control::Ref{tru_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export tru_import

function tru_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.tru_import(control::Ref{tru_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export tru_reset_control_s

function tru_reset_control_s(control, data, status)
  @ccall libgalahad_single.tru_reset_control_s(control::Ref{tru_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export tru_reset_control

function tru_reset_control(control, data, status)
  @ccall libgalahad_double.tru_reset_control(control::Ref{tru_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export tru_solve_with_mat_s

function tru_solve_with_mat_s(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_prec)
  @ccall libgalahad_single.tru_solve_with_mat_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float32}, g::Ptr{Float32},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_with_mat

function tru_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                          eval_prec)
  @ccall libgalahad_double.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint,
                                              x::Ptr{Float64}, g::Ptr{Float64},
                                              ne::Cint, eval_f::Ptr{Cvoid},
                                              eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_without_mat_s

function tru_solve_without_mat_s(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_prec)
  @ccall libgalahad_single.tru_solve_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float32},
                                                   g::Ptr{Float32}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_without_mat

function tru_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                             eval_prec)
  @ccall libgalahad_double.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                 n::Cint, x::Ptr{Float64},
                                                 g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_reverse_with_mat_s

function tru_solve_reverse_with_mat_s(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.tru_solve_reverse_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float32}, f::Float32,
                                                        g::Ptr{Float32}, ne::Cint,
                                                        H_val::Ptr{Float32},
                                                        u::Ptr{Float32},
                                                        v::Ptr{Float32})::Cvoid
end

export tru_solve_reverse_with_mat

function tru_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint},
                                                      eval_status::Ptr{Cint}, n::Cint,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Cint,
                                                      H_val::Ptr{Float64},
                                                      u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

export tru_solve_reverse_without_mat_s

function tru_solve_reverse_without_mat_s(data, status, eval_status, n, x, f, g, u, v)
  @ccall libgalahad_single.tru_solve_reverse_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float32}, f::Float32,
                                                           g::Ptr{Float32},
                                                           u::Ptr{Float32},
                                                           v::Ptr{Float32})::Cvoid
end

export tru_solve_reverse_without_mat

function tru_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v)
  @ccall libgalahad_double.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Cint},
                                                         eval_status::Ptr{Cint}, n::Cint,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64},
                                                         u::Ptr{Float64},
                                                         v::Ptr{Float64})::Cvoid
end

export tru_information_s

function tru_information_s(data, inform, status)
  @ccall libgalahad_single.tru_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{tru_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export tru_information

function tru_information(data, inform, status)
  @ccall libgalahad_double.tru_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{tru_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export tru_terminate_s

function tru_terminate_s(data, control, inform)
  @ccall libgalahad_single.tru_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{tru_control_type{Float32}},
                                           inform::Ref{tru_inform_type{Float32}})::Cvoid
end

export tru_terminate

function tru_terminate(data, control, inform)
  @ccall libgalahad_double.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{tru_control_type{Float64}},
                                         inform::Ref{tru_inform_type{Float64}})::Cvoid
end
