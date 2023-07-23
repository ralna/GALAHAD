export bgo_control_type

mutable struct bgo_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  attempts_max::Cint
  max_evals::Cint
  sampling_strategy::Cint
  hypercube_discretization::Cint
  alive_unit::Cint
  alive_file::NTuple{31,Cchar}
  infinity::T
  obj_unbounded::T
  cpu_time_limit::T
  clock_time_limit::T
  random_multistart::Bool
  hessian_available::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  ugo_control::ugo_control_type{T}
  lhs_control::lhs_control_type
  trb_control::trb_control_type{T}
  bgo_control_type{T}() where T = new()
end

export bgo_time_type

mutable struct bgo_time_type{T}
  total::Float32
  univariate_global::Float32
  multivariate_local::Float32
  clock_total::T
  clock_univariate_global::T
  clock_multivariate_local::T
  bgo_time_type{T}() where T = new()
end

export bgo_inform_type

mutable struct bgo_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  f_eval::Cint
  g_eval::Cint
  h_eval::Cint
  obj::T
  norm_pg::T
  time::bgo_time_type{T}
  ugo_inform::ugo_inform_type{T}
  lhs_inform::lhs_inform_type
  trb_inform::trb_inform_type{T}
  bgo_inform_type{T}() where T = new()
end

export bgo_initialize_s

function bgo_initialize_s(data, control, status)
  @ccall libgalahad_single.bgo_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{bgo_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export bgo_initialize

function bgo_initialize(data, control, status)
  @ccall libgalahad_double.bgo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{bgo_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export bgo_read_specfile_s

function bgo_read_specfile_s(control, specfile)
  @ccall libgalahad_single.bgo_read_specfile_s(control::Ref{bgo_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export bgo_read_specfile

function bgo_read_specfile(control, specfile)
  @ccall libgalahad_double.bgo_read_specfile(control::Ref{bgo_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export bgo_import_s

function bgo_import_s(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.bgo_import_s(control::Ref{bgo_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bgo_import

function bgo_import(control, data, status, n, x_l, x_u, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.bgo_import(control::Ref{bgo_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                      H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bgo_reset_control_s

function bgo_reset_control_s(control, data, status)
  @ccall libgalahad_single.bgo_reset_control_s(control::Ref{bgo_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export bgo_reset_control

function bgo_reset_control(control, data, status)
  @ccall libgalahad_double.bgo_reset_control(control::Ref{bgo_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export bgo_solve_with_mat_s

function bgo_solve_with_mat_s(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                            eval_hprod, eval_prec)
  @ccall libgalahad_single.bgo_solve_with_mat_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint,
                                                x::Ptr{Float32}, g::Ptr{Float32},
                                                ne::Cint, eval_f::Ptr{Cvoid},
                                                eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                eval_hprod::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_with_mat

function bgo_solve_with_mat(data, userdata, status, n, x, g, ne, eval_f, eval_g, eval_h,
                          eval_hprod, eval_prec)
  @ccall libgalahad_double.bgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint,
                                              x::Ptr{Float64}, g::Ptr{Float64},
                                              ne::Cint, eval_f::Ptr{Cvoid},
                                              eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                              eval_hprod::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_without_mat_s

function bgo_solve_without_mat_s(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                               eval_shprod, eval_prec)
  @ccall libgalahad_single.bgo_solve_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, x::Ptr{Float32},
                                                   g::Ptr{Float32}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_shprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_without_mat

function bgo_solve_without_mat(data, userdata, status, n, x, g, eval_f, eval_g, eval_hprod,
                             eval_shprod, eval_prec)
  @ccall libgalahad_double.bgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                 n::Cint, x::Ptr{Float64},
                                                 g::Ptr{Float64}, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_reverse_with_mat_s

function bgo_solve_reverse_with_mat_s(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.bgo_solve_reverse_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint},
                                                        eval_status::Ptr{Cint}, n::Cint,
                                                        x::Ptr{Float32}, f::Float32,
                                                        g::Ptr{Float32}, ne::Cint,
                                                        H_val::Ptr{Float32},
                                                        u::Ptr{Float32},
                                                        v::Ptr{Float32})::Cvoid
end

export bgo_solve_reverse_with_mat

function bgo_solve_reverse_with_mat(data, status, eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.bgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint},
                                                      eval_status::Ptr{Cint}, n::Cint,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Cint,
                                                      H_val::Ptr{Float64},
                                                      u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

export bgo_solve_reverse_without_mat_s

function bgo_solve_reverse_without_mat_s(data, status, eval_status, n, x, f, g, u, v,
                                       index_nz_v, nnz_v, index_nz_u, nnz_u)
  @ccall libgalahad_single.bgo_solve_reverse_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint},
                                                           eval_status::Ptr{Cint}, n::Cint,
                                                           x::Ptr{Float32}, f::Float32,
                                                           g::Ptr{Float32},
                                                           u::Ptr{Float32},
                                                           v::Ptr{Float32},
                                                           index_nz_v::Ptr{Cint},
                                                           nnz_v::Ptr{Cint},
                                                           index_nz_u::Ptr{Cint},
                                                           nnz_u::Cint)::Cvoid
end

export bgo_solve_reverse_without_mat

function bgo_solve_reverse_without_mat(data, status, eval_status, n, x, f, g, u, v,
                                     index_nz_v, nnz_v, index_nz_u, nnz_u)
  @ccall libgalahad_double.bgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Cint},
                                                         eval_status::Ptr{Cint}, n::Cint,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64},
                                                         u::Ptr{Float64},
                                                         v::Ptr{Float64},
                                                         index_nz_v::Ptr{Cint},
                                                         nnz_v::Ptr{Cint},
                                                         index_nz_u::Ptr{Cint},
                                                         nnz_u::Cint)::Cvoid
end

export bgo_information_s

function bgo_information_s(data, inform, status)
  @ccall libgalahad_single.bgo_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{bgo_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export bgo_information

function bgo_information(data, inform, status)
  @ccall libgalahad_double.bgo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{bgo_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export bgo_terminate_s

function bgo_terminate_s(data, control, inform)
  @ccall libgalahad_single.bgo_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{bgo_control_type{Float32}},
                                           inform::Ref{bgo_inform_type{Float32}})::Cvoid
end

export bgo_terminate

function bgo_terminate(data, control, inform)
  @ccall libgalahad_double.bgo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{bgo_control_type{Float64}},
                                         inform::Ref{bgo_inform_type{Float64}})::Cvoid
end
