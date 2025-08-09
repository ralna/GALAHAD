export bgo_control_type

struct bgo_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  attempts_max::INT
  max_evals::INT
  sampling_strategy::INT
  hypercube_discretization::INT
  alive_unit::INT
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
  ugo_control::ugo_control_type{T,INT}
  lhs_control::lhs_control_type{INT}
  trb_control::trb_control_type{T,INT}
end

export bgo_time_type

struct bgo_time_type{T}
  total::Float32
  univariate_global::Float32
  multivariate_local::Float32
  clock_total::T
  clock_univariate_global::T
  clock_multivariate_local::T
end

export bgo_inform_type

struct bgo_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  f_eval::INT
  g_eval::INT
  h_eval::INT
  obj::T
  norm_pg::T
  time::bgo_time_type{T}
  ugo_inform::ugo_inform_type{T,INT}
  lhs_inform::lhs_inform_type{INT}
  trb_inform::trb_inform_type{T,INT}
end

export bgo_initialize

function bgo_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.bgo_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bgo_control_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bgo_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.bgo_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{bgo_control_type{Float32,
                                                                                Int64}},
                                                  status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bgo_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.bgo_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bgo_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bgo_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.bgo_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bgo_control_type{Float64,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bgo_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.bgo_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{bgo_control_type{Float128,
                                                                             Int32}},
                                               status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bgo_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.bgo_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{bgo_control_type{Float128,
                                                                                   Int64}},
                                                     status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export bgo_read_specfile

function bgo_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.bgo_read_specfile_s(control::Ptr{bgo_control_type{Float32,Int32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function bgo_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.bgo_read_specfile_s_64(control::Ptr{bgo_control_type{Float32,
                                                                                   Int64}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function bgo_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.bgo_read_specfile(control::Ptr{bgo_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function bgo_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.bgo_read_specfile_64(control::Ptr{bgo_control_type{Float64,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function bgo_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.bgo_read_specfile_q(control::Ptr{bgo_control_type{Float128,
                                                                                Int32}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

function bgo_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.bgo_read_specfile_q_64(control::Ptr{bgo_control_type{Float128,
                                                                                      Int64}},
                                                        specfile::Ptr{Cchar})::Cvoid
end

export bgo_import

function bgo_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.bgo_import_s(control::Ptr{bgo_control_type{Float32,Int32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                        H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                        H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function bgo_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.bgo_import_s_64(control::Ptr{bgo_control_type{Float32,Int64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, x_l::Ptr{Float32},
                                              x_u::Ptr{Float32}, H_type::Ptr{Cchar},
                                              ne::Int64, H_row::Ptr{Int64},
                                              H_col::Ptr{Int64}, H_ptr::Ptr{Int64})::Cvoid
end

function bgo_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.bgo_import(control::Ptr{bgo_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function bgo_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.bgo_import_64(control::Ptr{bgo_control_type{Float64,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                            H_type::Ptr{Cchar}, ne::Int64,
                                            H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

function bgo_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.bgo_import_q(control::Ptr{bgo_control_type{Float128,Int32}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                           H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                           H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function bgo_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, x_l, x_u,
                    H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.bgo_import_q_64(control::Ptr{bgo_control_type{Float128,
                                                                               Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, x_l::Ptr{Float128},
                                                 x_u::Ptr{Float128}, H_type::Ptr{Cchar},
                                                 ne::Int64, H_row::Ptr{Int64},
                                                 H_col::Ptr{Int64},
                                                 H_ptr::Ptr{Int64})::Cvoid
end

export bgo_reset_control

function bgo_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.bgo_reset_control_s(control::Ptr{bgo_control_type{Float32,Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function bgo_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.bgo_reset_control_s_64(control::Ptr{bgo_control_type{Float32,
                                                                                   Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

function bgo_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.bgo_reset_control(control::Ptr{bgo_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function bgo_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.bgo_reset_control_64(control::Ptr{bgo_control_type{Float64,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

function bgo_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.bgo_reset_control_q(control::Ptr{bgo_control_type{Float128,
                                                                                Int32}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int32})::Cvoid
end

function bgo_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.bgo_reset_control_q_64(control::Ptr{bgo_control_type{Float128,
                                                                                      Int64}},
                                                        data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64})::Cvoid
end

export bgo_solve_with_mat

function bgo_solve_with_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_single.bgo_solve_with_mat_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Int32}, n::Int32,
                                                x::Ptr{Float32}, g::Ptr{Float32}, ne::Int32,
                                                eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_with_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_single_64.bgo_solve_with_mat_s_64(data::Ptr{Ptr{Cvoid}},
                                                      userdata::Ptr{Cvoid},
                                                      status::Ptr{Int64}, n::Int64,
                                                      x::Ptr{Float32}, g::Ptr{Float32},
                                                      ne::Int64, eval_f::Ptr{Cvoid},
                                                      eval_g::Ptr{Cvoid},
                                                      eval_h::Ptr{Cvoid},
                                                      eval_hprod::Ptr{Cvoid},
                                                      eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_with_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_double.bgo_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float64},
                                              g::Ptr{Float64}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid}, eval_hprod::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_with_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_double_64.bgo_solve_with_mat_64(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float64}, g::Ptr{Float64},
                                                    ne::Int64, eval_f::Ptr{Cvoid},
                                                    eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_with_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple.bgo_solve_with_mat_q(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                   n::Int32, x::Ptr{Float128},
                                                   g::Ptr{Float128}, ne::Int32,
                                                   eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                   eval_h::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_with_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple_64.bgo_solve_with_mat_q_64(data::Ptr{Ptr{Cvoid}},
                                                         userdata::Ptr{Cvoid},
                                                         status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float128}, g::Ptr{Float128},
                                                         ne::Int64, eval_f::Ptr{Cvoid},
                                                         eval_g::Ptr{Cvoid},
                                                         eval_h::Ptr{Cvoid},
                                                         eval_hprod::Ptr{Cvoid},
                                                         eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_without_mat

function bgo_solve_without_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single.bgo_solve_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                   n::Int32, x::Ptr{Float32},
                                                   g::Ptr{Float32}, eval_f::Ptr{Cvoid},
                                                   eval_g::Ptr{Cvoid},
                                                   eval_hprod::Ptr{Cvoid},
                                                   eval_shprod::Ptr{Cvoid},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_single_64.bgo_solve_without_mat_s_64(data::Ptr{Ptr{Cvoid}},
                                                         userdata::Ptr{Cvoid},
                                                         status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float32}, g::Ptr{Float32},
                                                         eval_f::Ptr{Cvoid},
                                                         eval_g::Ptr{Cvoid},
                                                         eval_hprod::Ptr{Cvoid},
                                                         eval_shprod::Ptr{Cvoid},
                                                         eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double.bgo_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float64}, g::Ptr{Float64},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_shprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_double_64.bgo_solve_without_mat_64(data::Ptr{Ptr{Cvoid}},
                                                       userdata::Ptr{Cvoid},
                                                       status::Ptr{Int64}, n::Int64,
                                                       x::Ptr{Float64}, g::Ptr{Float64},
                                                       eval_f::Ptr{Cvoid},
                                                       eval_g::Ptr{Cvoid},
                                                       eval_hprod::Ptr{Cvoid},
                                                       eval_shprod::Ptr{Cvoid},
                                                       eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple.bgo_solve_without_mat_q(data::Ptr{Ptr{Cvoid}},
                                                      userdata::Ptr{Cvoid},
                                                      status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float128}, g::Ptr{Float128},
                                                      eval_f::Ptr{Cvoid},
                                                      eval_g::Ptr{Cvoid},
                                                      eval_hprod::Ptr{Cvoid},
                                                      eval_shprod::Ptr{Cvoid},
                                                      eval_prec::Ptr{Cvoid})::Cvoid
end

function bgo_solve_without_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_shprod, eval_prec)
  @ccall libgalahad_quadruple_64.bgo_solve_without_mat_q_64(data::Ptr{Ptr{Cvoid}},
                                                            userdata::Ptr{Cvoid},
                                                            status::Ptr{Int64}, n::Int64,
                                                            x::Ptr{Float128},
                                                            g::Ptr{Float128},
                                                            eval_f::Ptr{Cvoid},
                                                            eval_g::Ptr{Cvoid},
                                                            eval_hprod::Ptr{Cvoid},
                                                            eval_shprod::Ptr{Cvoid},
                                                            eval_prec::Ptr{Cvoid})::Cvoid
end

export bgo_solve_reverse_with_mat

function bgo_solve_reverse_with_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.bgo_solve_reverse_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32},
                                                        eval_status::Ptr{Int32}, n::Int32,
                                                        x::Ptr{Float32}, f::Float32,
                                                        g::Ptr{Float32}, ne::Int32,
                                                        H_val::Ptr{Float32},
                                                        u::Ptr{Float32},
                                                        v::Ptr{Float32})::Cvoid
end

function bgo_solve_reverse_with_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single_64.bgo_solve_reverse_with_mat_s_64(data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int64},
                                                              eval_status::Ptr{Int64},
                                                              n::Int64, x::Ptr{Float32},
                                                              f::Float32, g::Ptr{Float32},
                                                              ne::Int64,
                                                              H_val::Ptr{Float32},
                                                              u::Ptr{Float32},
                                                              v::Ptr{Float32})::Cvoid
end

function bgo_solve_reverse_with_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.bgo_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Int32,
                                                      H_val::Ptr{Float64}, u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

function bgo_solve_reverse_with_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double_64.bgo_solve_reverse_with_mat_64(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float64},
                                                            f::Float64, g::Ptr{Float64},
                                                            ne::Int64, H_val::Ptr{Float64},
                                                            u::Ptr{Float64},
                                                            v::Ptr{Float64})::Cvoid
end

function bgo_solve_reverse_with_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple.bgo_solve_reverse_with_mat_q(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int32},
                                                           eval_status::Ptr{Int32},
                                                           n::Int32, x::Ptr{Float128},
                                                           f::Cfloat128, g::Ptr{Float128},
                                                           ne::Int32, H_val::Ptr{Float128},
                                                           u::Ptr{Float128},
                                                           v::Ptr{Float128})::Cvoid
end

function bgo_solve_reverse_with_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple_64.bgo_solve_reverse_with_mat_q_64(data::Ptr{Ptr{Cvoid}},
                                                                 status::Ptr{Int64},
                                                                 eval_status::Ptr{Int64},
                                                                 n::Int64, x::Ptr{Float128},
                                                                 f::Cfloat128,
                                                                 g::Ptr{Float128},
                                                                 ne::Int64,
                                                                 H_val::Ptr{Float128},
                                                                 u::Ptr{Float128},
                                                                 v::Ptr{Float128})::Cvoid
end

export bgo_solve_reverse_without_mat

function bgo_solve_reverse_without_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single.bgo_solve_reverse_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int32},
                                                           eval_status::Ptr{Int32},
                                                           n::Int32, x::Ptr{Float32},
                                                           f::Float32, g::Ptr{Float32},
                                                           u::Ptr{Float32}, v::Ptr{Float32},
                                                           index_nz_v::Ptr{Int32},
                                                           nnz_v::Ptr{Int32},
                                                           index_nz_u::Ptr{Int32},
                                                           nnz_u::Int32)::Cvoid
end

function bgo_solve_reverse_without_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_single_64.bgo_solve_reverse_without_mat_s_64(data::Ptr{Ptr{Cvoid}},
                                                                 status::Ptr{Int64},
                                                                 eval_status::Ptr{Int64},
                                                                 n::Int64, x::Ptr{Float32},
                                                                 f::Float32,
                                                                 g::Ptr{Float32},
                                                                 u::Ptr{Float32},
                                                                 v::Ptr{Float32},
                                                                 index_nz_v::Ptr{Int64},
                                                                 nnz_v::Ptr{Int64},
                                                                 index_nz_u::Ptr{Int64},
                                                                 nnz_u::Int64)::Cvoid
end

function bgo_solve_reverse_without_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double.bgo_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
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

function bgo_solve_reverse_without_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_double_64.bgo_solve_reverse_without_mat_64(data::Ptr{Ptr{Cvoid}},
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

function bgo_solve_reverse_without_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple.bgo_solve_reverse_without_mat_q(data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int32},
                                                              eval_status::Ptr{Int32},
                                                              n::Int32, x::Ptr{Float128},
                                                              f::Cfloat128,
                                                              g::Ptr{Float128},
                                                              u::Ptr{Float128},
                                                              v::Ptr{Float128},
                                                              index_nz_v::Ptr{Int32},
                                                              nnz_v::Ptr{Int32},
                                                              index_nz_u::Ptr{Int32},
                                                              nnz_u::Int32)::Cvoid
end

function bgo_solve_reverse_without_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v, index_nz_v, nnz_v,
                                       index_nz_u, nnz_u)
  @ccall libgalahad_quadruple_64.bgo_solve_reverse_without_mat_q_64(data::Ptr{Ptr{Cvoid}},
                                                                    status::Ptr{Int64},
                                                                    eval_status::Ptr{Int64},
                                                                    n::Int64,
                                                                    x::Ptr{Float128},
                                                                    f::Cfloat128,
                                                                    g::Ptr{Float128},
                                                                    u::Ptr{Float128},
                                                                    v::Ptr{Float128},
                                                                    index_nz_v::Ptr{Int64},
                                                                    nnz_v::Ptr{Int64},
                                                                    index_nz_u::Ptr{Int64},
                                                                    nnz_u::Int64)::Cvoid
end

export bgo_information

function bgo_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.bgo_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{bgo_inform_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function bgo_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.bgo_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{bgo_inform_type{Float32,
                                                                               Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function bgo_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.bgo_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bgo_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bgo_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.bgo_information_64(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{bgo_inform_type{Float64,Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function bgo_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.bgo_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{bgo_inform_type{Float128,Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function bgo_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.bgo_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{bgo_inform_type{Float128,
                                                                                  Int64}},
                                                      status::Ptr{Int64})::Cvoid
end

export bgo_terminate

function bgo_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bgo_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bgo_control_type{Float32,Int32}},
                                           inform::Ptr{bgo_inform_type{Float32,Int32}})::Cvoid
end

function bgo_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bgo_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{bgo_control_type{Float32,
                                                                               Int64}},
                                                 inform::Ptr{bgo_inform_type{Float32,Int64}})::Cvoid
end

function bgo_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bgo_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bgo_control_type{Float64,Int32}},
                                         inform::Ptr{bgo_inform_type{Float64,Int32}})::Cvoid
end

function bgo_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bgo_terminate_64(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{bgo_control_type{Float64,Int64}},
                                               inform::Ptr{bgo_inform_type{Float64,Int64}})::Cvoid
end

function bgo_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bgo_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{bgo_control_type{Float128,Int32}},
                                              inform::Ptr{bgo_inform_type{Float128,Int32}})::Cvoid
end

function bgo_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bgo_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{bgo_control_type{Float128,
                                                                                  Int64}},
                                                    inform::Ptr{bgo_inform_type{Float128,
                                                                                Int64}})::Cvoid
end

function run_sif(::Val{:bgo}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runbgo_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:bgo}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runbgo_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
