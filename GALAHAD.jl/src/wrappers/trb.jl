export trb_control_type

struct trb_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  alive_unit::INT
  alive_file::NTuple{31,Cchar}
  more_toraldo::INT
  non_monotone::INT
  model::INT
  norm::INT
  semi_bandwidth::INT
  lbfgs_vectors::INT
  max_dxg::INT
  icfs_vectors::INT
  mi28_lsize::INT
  mi28_rsize::INT
  advanced_start::INT
  infinity::T
  stop_pg_absolute::T
  stop_pg_relative::T
  stop_s::T
  initial_radius::T
  maximum_radius::T
  stop_rel_cg::T
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
  two_norm_tr::Bool
  exact_gcp::Bool
  accurate_bqp::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  trs_control::trs_control_type{T,INT}
  gltr_control::gltr_control_type{T,INT}
  psls_control::psls_control_type{T,INT}
  lms_control::lms_control_type{INT}
  lms_control_prec::lms_control_type{INT}
  sha_control::sha_control_type{INT}
end

export trb_time_type

struct trb_time_type{T}
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
end

export trb_inform_type

struct trb_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  cg_iter::INT
  cg_maxit::INT
  f_eval::INT
  g_eval::INT
  h_eval::INT
  n_free::INT
  factorization_status::INT
  factorization_max::INT
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  obj::T
  norm_pg::T
  radius::T
  time::trb_time_type{T}
  trs_inform::trs_inform_type{T,INT}
  gltr_inform::gltr_inform_type{T,INT}
  psls_inform::psls_inform_type{T,INT}
  lms_inform::lms_inform_type{T,INT}
  lms_inform_prec::lms_inform_type{T,INT}
  sha_inform::sha_inform_type{T,INT}
end

export trb_initialize

function trb_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{trb_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function trb_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{trb_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function trb_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{trb_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function trb_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{trb_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function trb_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{trb_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function trb_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.trb_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{trb_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export trb_read_specfile

function trb_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.trb_read_specfile(control::Ptr{trb_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function trb_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.trb_read_specfile(control::Ptr{trb_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function trb_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.trb_read_specfile(control::Ptr{trb_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function trb_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.trb_read_specfile(control::Ptr{trb_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function trb_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.trb_read_specfile(control::Ptr{trb_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function trb_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.trb_read_specfile(control::Ptr{trb_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export trb_import

function trb_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.trb_import(control::Ptr{trb_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function trb_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.trb_import(control::Ptr{trb_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                         H_type::Ptr{Cchar}, ne::Int64, H_row::Ptr{Int64},
                                         H_col::Ptr{Int64}, H_ptr::Ptr{Int64})::Cvoid
end

function trb_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.trb_import(control::Ptr{trb_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function trb_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.trb_import(control::Ptr{trb_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                         H_type::Ptr{Cchar}, ne::Int64, H_row::Ptr{Int64},
                                         H_col::Ptr{Int64}, H_ptr::Ptr{Int64})::Cvoid
end

function trb_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.trb_import(control::Ptr{trb_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                         H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                         H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function trb_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.trb_import(control::Ptr{trb_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, x_l::Ptr{Float128},
                                            x_u::Ptr{Float128}, H_type::Ptr{Cchar},
                                            ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

export trb_reset_control

function trb_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.trb_reset_control(control::Ptr{trb_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function trb_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.trb_reset_control(control::Ptr{trb_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function trb_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.trb_reset_control(control::Ptr{trb_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function trb_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.trb_reset_control(control::Ptr{trb_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function trb_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.trb_reset_control(control::Ptr{trb_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function trb_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.trb_reset_control(control::Ptr{trb_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export trb_solve_with_mat

function trb_solve_with_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_single.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float32},
                                              g::Ptr{Float32}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_with_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_single_64.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float32}, g::Ptr{Float32},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_with_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_double.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float64},
                                              g::Ptr{Float64}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_with_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_double_64.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float64}, g::Ptr{Float64},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_with_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_quadruple.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float128},
                                                 g::Ptr{Float128}, ne::Int32,
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_with_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_quadruple_64.trb_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    ne::Int64, eval_f::Ptr{Cvoid},
                                                    eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

export trb_solve_without_mat

function trb_solve_without_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float32}, g::Ptr{Float32},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single_64.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float32}, g::Ptr{Float32},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float64}, g::Ptr{Float64},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double_64.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float64}, g::Ptr{Float64},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int32}, n::Int32,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function trb_solve_without_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple_64.trb_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                       userdata::Ptr{Cvoid},
                                                       status::Ptr{Int64}, n::Int64,
                                                       x::Ptr{Float128}, g::Ptr{Float128},
                                                       eval_f::Ptr{Cvoid},
                                                       eval_g::Ptr{Cvoid},
                                                       eval_hprod::Ptr{Cvoid},
                                                       eval_shprod::Ptr{Cvoid},
                                                       eval_prec::Ptr{Cvoid})::Cvoid
end

export trb_solve_reverse_with_mat

function trb_solve_reverse_with_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float32}, f::Float32,
                                                      g::Ptr{Float32}, ne::Int32,
                                                      H_val::Ptr{Float32}, u::Ptr{Float32},
                                                      v::Ptr{Float32})::Cvoid
end

function trb_solve_reverse_with_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single_64.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float32}, f::Float32,
                                                         g::Ptr{Float32}, ne::Int64,
                                                         H_val::Ptr{Float32},
                                                         u::Ptr{Float32},
                                                         v::Ptr{Float32})::Cvoid
end

function trb_solve_reverse_with_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Int32,
                                                      H_val::Ptr{Float64}, u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

function trb_solve_reverse_with_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double_64.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, ne::Int64,
                                                         H_val::Ptr{Float64},
                                                         u::Ptr{Float64},
                                                         v::Ptr{Float64})::Cvoid
end

function trb_solve_reverse_with_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float128}, f::Cfloat128,
                                                         g::Ptr{Float128}, ne::Int32,
                                                         H_val::Ptr{Float128},
                                                         u::Ptr{Float128},
                                                         v::Ptr{Float128})::Cvoid
end

function trb_solve_reverse_with_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple_64.trb_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float128},
                                                            f::Cfloat128, g::Ptr{Float128},
                                                            ne::Int64, H_val::Ptr{Float128},
                                                            u::Ptr{Float128},
                                                            v::Ptr{Float128})::Cvoid
end

export trb_solve_reverse_without_mat

function trb_solve_reverse_without_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float32}, f::Float32,
                                                         g::Ptr{Float32}, u::Ptr{Float32},
                                                         v::Ptr{Float32},
                                                         index_nz_v::Ptr{Int32},
                                                         nnz_v::Ptr{Int32},
                                                         index_nz_u::Ptr{Int32},
                                                         nnz_u::Int32)::Cvoid
end

function trb_solve_reverse_without_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single_64.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float32},
                                                            f::Float32, g::Ptr{Float32},
                                                            u::Ptr{Float32},
                                                            v::Ptr{Float32},
                                                            index_nz_v::Ptr{Int64},
                                                            nnz_v::Ptr{Int64},
                                                            index_nz_u::Ptr{Int64},
                                                            nnz_u::Int64)::Cvoid
end

function trb_solve_reverse_without_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, u::Ptr{Float64},
                                                         v::Ptr{Float64},
                                                         index_nz_v::Ptr{Int32},
                                                         nnz_v::Ptr{Int32},
                                                         index_nz_u::Ptr{Int32},
                                                         nnz_u::Int32)::Cvoid
end

function trb_solve_reverse_without_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double_64.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float64},
                                                            f::Float64, g::Ptr{Float64},
                                                            u::Ptr{Float64},
                                                            v::Ptr{Float64},
                                                            index_nz_v::Ptr{Int64},
                                                            nnz_v::Ptr{Int64},
                                                            index_nz_u::Ptr{Int64},
                                                            nnz_u::Int64)::Cvoid
end

function trb_solve_reverse_without_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int32},
                                                            eval_status::Ptr{Int32},
                                                            n::Int32, x::Ptr{Float128},
                                                            f::Cfloat128, g::Ptr{Float128},
                                                            u::Ptr{Float128},
                                                            v::Ptr{Float128},
                                                            index_nz_v::Ptr{Int32},
                                                            nnz_v::Ptr{Int32},
                                                            index_nz_u::Ptr{Int32},
                                                            nnz_u::Int32)::Cvoid
end

function trb_solve_reverse_without_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple_64.trb_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                               status::Ptr{Int64},
                                                               eval_status::Ptr{Int64},
                                                               n::Int64, x::Ptr{Float128},
                                                               f::Cfloat128,
                                                               g::Ptr{Float128},
                                                               u::Ptr{Float128},
                                                               v::Ptr{Float128},
                                                               index_nz_v::Ptr{Int64},
                                                               nnz_v::Ptr{Int64},
                                                               index_nz_u::Ptr{Int64},
                                                               nnz_u::Int64)::Cvoid
end

export trb_information

function trb_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.trb_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{trb_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function trb_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.trb_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{trb_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function trb_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.trb_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{trb_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function trb_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.trb_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{trb_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function trb_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.trb_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{trb_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function trb_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.trb_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{trb_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export trb_terminate

function trb_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{trb_control_type{Float32,Int32}},
                                         inform::Ptr{trb_inform_type{Float32,Int32}})::Cvoid
end

function trb_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trb_control_type{Float32,Int64}},
                                            inform::Ptr{trb_inform_type{Float32,Int64}})::Cvoid
end

function trb_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{trb_control_type{Float64,Int32}},
                                         inform::Ptr{trb_inform_type{Float64,Int32}})::Cvoid
end

function trb_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trb_control_type{Float64,Int64}},
                                            inform::Ptr{trb_inform_type{Float64,Int64}})::Cvoid
end

function trb_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trb_control_type{Float128,Int32}},
                                            inform::Ptr{trb_inform_type{Float128,Int32}})::Cvoid
end

function trb_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.trb_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{trb_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{trb_inform_type{Float128,Int64}})::Cvoid
end
