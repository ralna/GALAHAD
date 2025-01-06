export dgo_control_type

struct dgo_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  max_evals::INT
  dictionary_size::INT
  alive_unit::INT
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
  hash_control::hash_control_type{INT}
  ugo_control::ugo_control_type{T,INT}
  trb_control::trb_control_type{T,INT}
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

struct dgo_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  f_eval::INT
  g_eval::INT
  h_eval::INT
  obj::T
  norm_pg::T
  length_ratio::T
  f_gap::T
  why_stop::NTuple{2,Cchar}
  time::dgo_time_type{T}
  hash_inform::hash_inform_type{INT}
  ugo_inform::ugo_inform_type{T,INT}
  trb_inform::trb_inform_type{T,INT}
end

export dgo_initialize

function dgo_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dgo_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dgo_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dgo_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dgo_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{dgo_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function dgo_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dgo_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function dgo_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{dgo_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function dgo_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.dgo_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{dgo_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export dgo_read_specfile

function dgo_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.dgo_read_specfile(control::Ptr{dgo_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dgo_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.dgo_read_specfile(control::Ptr{dgo_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dgo_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.dgo_read_specfile(control::Ptr{dgo_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function dgo_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.dgo_read_specfile(control::Ptr{dgo_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dgo_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.dgo_read_specfile(control::Ptr{dgo_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function dgo_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.dgo_read_specfile(control::Ptr{dgo_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export dgo_import

function dgo_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.dgo_import(control::Ptr{dgo_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function dgo_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.dgo_import(control::Ptr{dgo_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                         H_type::Ptr{Cchar}, ne::Int64, H_row::Ptr{Int64},
                                         H_col::Ptr{Int64}, H_ptr::Ptr{Int64})::Cvoid
end

function dgo_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.dgo_import(control::Ptr{dgo_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function dgo_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.dgo_import(control::Ptr{dgo_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                         H_type::Ptr{Cchar}, ne::Int64, H_row::Ptr{Int64},
                                         H_col::Ptr{Int64}, H_ptr::Ptr{Int64})::Cvoid
end

function dgo_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.dgo_import(control::Ptr{dgo_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                         H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                         H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function dgo_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.dgo_import(control::Ptr{dgo_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, x_l::Ptr{Float128},
                                            x_u::Ptr{Float128}, H_type::Ptr{Cchar},
                                            ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

export dgo_reset_control

function dgo_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.dgo_reset_control(control::Ptr{dgo_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dgo_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.dgo_reset_control(control::Ptr{dgo_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dgo_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.dgo_reset_control(control::Ptr{dgo_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function dgo_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.dgo_reset_control(control::Ptr{dgo_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function dgo_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.dgo_reset_control(control::Ptr{dgo_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function dgo_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.dgo_reset_control(control::Ptr{dgo_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export dgo_solve_with_mat

function dgo_solve_with_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_single.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float32},
                                              g::Ptr{Float32}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_with_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_single_64.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float32}, g::Ptr{Float32},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_with_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_double.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float64},
                                              g::Ptr{Float64}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_with_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_double_64.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float64}, g::Ptr{Float64},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_with_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float128},
                                                 g::Ptr{Float128}, ne::Int32,
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_with_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple_64.dgo_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    ne::Int64, eval_f::Ptr{Cvoid},
                                                    eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_without_mat

function dgo_solve_without_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float32}, g::Ptr{Float32},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_without_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single_64.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float32}, g::Ptr{Float32},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_without_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float64}, g::Ptr{Float64},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_without_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double_64.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float64}, g::Ptr{Float64},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_without_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int32}, n::Int32,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_shprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function dgo_solve_without_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple_64.dgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                       userdata::Ptr{Cvoid},
                                                       status::Ptr{Int64}, n::Int64,
                                                       x::Ptr{Float128}, g::Ptr{Float128},
                                                       eval_f::Ptr{Cvoid},
                                                       eval_g::Ptr{Cvoid},
                                                       eval_hprod::Ptr{Cvoid},
                                                       eval_shprod::Ptr{Cvoid},
                                                       eval_prec::Ptr{Cvoid})::Cvoid
end

export dgo_solve_reverse_with_mat

function dgo_solve_reverse_with_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float32}, f::Float32,
                                                      g::Ptr{Float32}, ne::Int32,
                                                      H_val::Ptr{Float32}, u::Ptr{Float32},
                                                      v::Ptr{Float32})::Cvoid
end

function dgo_solve_reverse_with_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single_64.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float32}, f::Float32,
                                                         g::Ptr{Float32}, ne::Int64,
                                                         H_val::Ptr{Float32},
                                                         u::Ptr{Float32},
                                                         v::Ptr{Float32})::Cvoid
end

function dgo_solve_reverse_with_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Int32,
                                                      H_val::Ptr{Float64}, u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

function dgo_solve_reverse_with_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double_64.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, ne::Int64,
                                                         H_val::Ptr{Float64},
                                                         u::Ptr{Float64},
                                                         v::Ptr{Float64})::Cvoid
end

function dgo_solve_reverse_with_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float128}, f::Cfloat128,
                                                         g::Ptr{Float128}, ne::Int32,
                                                         H_val::Ptr{Float128},
                                                         u::Ptr{Float128},
                                                         v::Ptr{Float128})::Cvoid
end

function dgo_solve_reverse_with_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple_64.dgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float128},
                                                            f::Cfloat128, g::Ptr{Float128},
                                                            ne::Int64, H_val::Ptr{Float128},
                                                            u::Ptr{Float128},
                                                            v::Ptr{Float128})::Cvoid
end

export dgo_solve_reverse_without_mat

function dgo_solve_reverse_without_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function dgo_solve_reverse_without_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single_64.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function dgo_solve_reverse_without_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function dgo_solve_reverse_without_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double_64.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function dgo_solve_reverse_without_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function dgo_solve_reverse_without_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple_64.dgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

export dgo_information

function dgo_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.dgo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dgo_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dgo_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.dgo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dgo_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dgo_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.dgo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{dgo_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function dgo_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.dgo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dgo_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function dgo_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.dgo_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{dgo_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function dgo_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.dgo_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{dgo_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export dgo_terminate

function dgo_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dgo_control_type{Float32,Int32}},
                                         inform::Ptr{dgo_inform_type{Float32,Int32}})::Cvoid
end

function dgo_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dgo_control_type{Float32,Int64}},
                                            inform::Ptr{dgo_inform_type{Float32,Int64}})::Cvoid
end

function dgo_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{dgo_control_type{Float64,Int32}},
                                         inform::Ptr{dgo_inform_type{Float64,Int32}})::Cvoid
end

function dgo_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dgo_control_type{Float64,Int64}},
                                            inform::Ptr{dgo_inform_type{Float64,Int64}})::Cvoid
end

function dgo_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dgo_control_type{Float128,Int32}},
                                            inform::Ptr{dgo_inform_type{Float128,Int32}})::Cvoid
end

function dgo_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.dgo_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{dgo_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{dgo_inform_type{Float128,Int64}})::Cvoid
end
