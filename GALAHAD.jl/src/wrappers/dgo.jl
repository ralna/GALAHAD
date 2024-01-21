export dgo_control_type

struct dgo_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  max_evals::Cint
  dictionary_size::Cint
  alive_unit::Cint
  alive_file::NTuple{31,Cchar}
  infinity::T
  lipschitz_lower_bound::T
  lipschitz_reliability::T
  lipschitz_control::T
  stop_length::T
  stop_f::T
  obj_unbounded::T
  cpu_time_limit::T
  clock_time_limit::T
  hessian_available::Bool
  prune::Bool
  perform_local_optimization::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  hash_control::hash_control_type
  ugo_control::ugo_control_type{T}
  trb_control::trb_control_type{T}
end

export dgo_time_type

struct dgo_time_type{T}
  total::Float32
  univariate_global::Float32
  multivariate_local::Float32
  clock_total::T
  clock_univariate_global::T
  clock_multivariate_local::T
end

export dgo_inform_type

struct dgo_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  f_eval::Cint
  g_eval::Cint
  h_eval::Cint
  obj::T
  norm_pg::T
  length_ratio::T
  f_gap::T
  why_stop::NTuple{2,Cchar}
  time::dgo_time_type{T}
  hash_inform::hash_inform_type
  ugo_inform::ugo_inform_type{T}
  trb_inform::trb_inform_type{T}
end

export dgo_initialize_s

function dgo_initialize_s(data, control, status)
  @ccall libgalahad_single.dgo_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dgo_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export dgo_initialize

function dgo_initialize(data, control, status)
  @ccall libgalahad_double.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dgo_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export dgo_read_specfile_s

function dgo_read_specfile_s(control, specfile)
  @ccall libgalahad_single.dgo_read_specfile_s(control::Ptr{dgo_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export dgo_read_specfile

function dgo_read_specfile(control, specfile)
  @ccall libgalahad_double.dgo_read_specfile(control::Ptr{dgo_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export dgo_import_s

function dgo_import_s(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.dgo_import_s(control::Ptr{dgo_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export dgo_import

function dgo_import(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.dgo_import(control::Ptr{dgo_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                      H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export dgo_reset_control_s

function dgo_reset_control_s(control, data, status)
  @ccall libgalahad_single.dgo_reset_control_s(control::Ptr{dgo_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export dgo_reset_control

function dgo_reset_control(control, data, status)
  @ccall libgalahad_double.dgo_reset_control(control::Ptr{dgo_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export dgo_solve_with_mat_s

function dgo_solve_with_mat_s(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                              eval_hprod, eval_prec)
  @ccall libgalahad_single.dgo_solve_with_mat_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint, x::Ptr{Float32},
                                                g::Ptr{Float32}, ne::Cint,
                                                eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_with_mat

function dgo_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_hprod, eval_prec)
  @ccall libgalahad_double.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint, x::Ptr{Float64},
                                              g::Ptr{Float64}, ne::Cint, eval_f::Ptr{Cvoid},
                                              eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                              eval_hprod::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_without_mat_s

function dgo_solve_without_mat_s(data, userdata, status, n, x, g, eval_f, eval_g,
                                 eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single.dgo_solve_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float32},
                                                   g::Ptr{Float32}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_shprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_without_mat

function dgo_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_shprod, eval_prec)
  @ccall libgalahad_double.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                 n::Cint, x::Ptr{Float64}, g::Ptr{Float64},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_reverse_with_mat_s

function dgo_solve_reverse_with_mat_s(data, status, eval_status, n, x, f, g, ne, H_val, u,
                                      v)
  @ccall libgalahad_single.dgo_solve_reverse_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float32}, f::Float32,
                                                        g::Ptr{Float32}, ne::Cint,
                                                        H_val::Ptr{Float32},
                                                        u::Ptr{Float32},
                                                        v::Ptr{Float32})::Cvoid
end

export dgo_solve_reverse_with_mat

function dgo_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint},
                                                      eval_status::Ptr{Cint}, n::Cint,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Cint,
                                                      H_val::Ptr{Float64}, u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

export dgo_solve_reverse_without_mat_s

function dgo_solve_reverse_without_mat_s(data, status, eval_status, n, x, f, g, u, v,
                                         index_nz_v, nnz_v, index_nz_u, nnz_u)
  @ccall libgalahad_single.dgo_solve_reverse_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float32}, f::Float32,
                                                           g::Ptr{Float32}, u::Ptr{Float32},
                                                           v::Ptr{Float32},
                                                           index_nz_v::Ptr{Cint},
                                                           nnz_v::Ptr{Cint},
                                                           index_nz_u::Ptr{Cint},
                                                           nnz_u::Cint)::Cvoid
end

export dgo_solve_reverse_without_mat

function dgo_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v,
                                       index_nz_v, nnz_v, index_nz_u, nnz_u)
  @ccall libgalahad_double.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Cint},
                                                         eval_status::Ptr{Cint}, n::Cint,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, u::Ptr{Float64},
                                                         v::Ptr{Float64},
                                                         index_nz_v::Ptr{Cint},
                                                         nnz_v::Ptr{Cint},
                                                         index_nz_u::Ptr{Cint},
                                                         nnz_u::Cint)::Cvoid
end

export dgo_information_s

function dgo_information_s(data, inform, status)
  @ccall libgalahad_single.dgo_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{dgo_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export dgo_information

function dgo_information(data, inform, status)
  @ccall libgalahad_double.dgo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dgo_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export dgo_terminate_s

function dgo_terminate_s(data, control, inform)
  @ccall libgalahad_single.dgo_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{dgo_control_type{Float32}},
                                           inform::Ptr{dgo_inform_type{Float32}})::Cvoid
end

export dgo_terminate

function dgo_terminate(data, control, inform)
  @ccall libgalahad_double.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dgo_control_type{Float64}},
                                         inform::Ptr{dgo_inform_type{Float64}})::Cvoid
end
