export tru_control_type

struct tru_control_type{T,INT}
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
  stop_g_absolute::T
  stop_g_relative::T
  stop_s::T
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
  trs_control::trs_control_type{T,INT}
  gltr_control::gltr_control_type{T,INT}
  dps_control::dps_control_type{T,INT}
  psls_control::psls_control_type{T,INT}
  lms_control::lms_control_type{INT}
  lms_control_prec::lms_control_type{INT}
  sec_control::sec_control_type{T,INT}
  sha_control::sha_control_type{INT}
end

export tru_time_type

struct tru_time_type{T}
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

export tru_inform_type

struct tru_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  cg_iter::INT
  f_eval::INT
  g_eval::INT
  h_eval::INT
  factorization_max::INT
  factorization_status::INT
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_g::T
  radius::T
  time::tru_time_type{T}
  trs_inform::trs_inform_type{T,INT}
  gltr_inform::gltr_inform_type{T,INT}
  dps_inform::dps_inform_type{T,INT}
  psls_inform::psls_inform_type{T,INT}
  lms_inform::lms_inform_type{T,INT}
  lms_inform_prec::lms_inform_type{T,INT}
  sec_inform::sec_inform_type{INT}
  sha_inform::sha_inform_type{T,INT}
end

export tru_initialize

function tru_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{tru_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function tru_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{tru_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function tru_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{tru_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function tru_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{tru_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function tru_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{tru_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function tru_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.tru_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{tru_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export tru_read_specfile

function tru_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.tru_read_specfile(control::Ptr{tru_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function tru_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.tru_read_specfile(control::Ptr{tru_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function tru_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.tru_read_specfile(control::Ptr{tru_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function tru_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.tru_read_specfile(control::Ptr{tru_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function tru_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.tru_read_specfile(control::Ptr{tru_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function tru_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.tru_read_specfile(control::Ptr{tru_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export tru_import

function tru_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single.tru_import(control::Ptr{tru_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function tru_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.tru_import(control::Ptr{tru_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function tru_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double.tru_import(control::Ptr{tru_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function tru_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.tru_import(control::Ptr{tru_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function tru_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.tru_import(control::Ptr{tru_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, H_type::Ptr{Cchar}, ne::Int32,
                                         H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32})::Cvoid
end

function tru_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.tru_import(control::Ptr{tru_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                            H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

export tru_reset_control

function tru_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.tru_reset_control(control::Ptr{tru_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function tru_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.tru_reset_control(control::Ptr{tru_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function tru_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.tru_reset_control(control::Ptr{tru_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function tru_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.tru_reset_control(control::Ptr{tru_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function tru_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.tru_reset_control(control::Ptr{tru_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function tru_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.tru_reset_control(control::Ptr{tru_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export tru_solve_with_mat

function tru_solve_with_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_single.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float32},
                                              g::Ptr{Float32}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_with_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_single_64.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float32}, g::Ptr{Float32},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_with_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_double.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, x::Ptr{Float64},
                                              g::Ptr{Float64}, ne::Int32,
                                              eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                              eval_h::Ptr{Cvoid},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_with_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x, g,
                            ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_double_64.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, x::Ptr{Float64}, g::Ptr{Float64},
                                                 ne::Int64, eval_f::Ptr{Cvoid},
                                                 eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_with_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_quadruple.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float128},
                                                 g::Ptr{Float128}, ne::Int32,
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_h::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_with_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, x,
                            g, ne, eval_f, eval_g, eval_h, eval_prec)
  @ccall libgalahad_quadruple_64.tru_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    ne::Int64, eval_f::Ptr{Cvoid},
                                                    eval_g::Ptr{Cvoid}, eval_h::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_without_mat

function tru_solve_without_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_single.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float32}, g::Ptr{Float32},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_single_64.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float32}, g::Ptr{Float32},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_double.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, x::Ptr{Float64}, g::Ptr{Float64},
                                                 eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                 eval_hprod::Ptr{Cvoid},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, x,
                               g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_double_64.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64,
                                                    x::Ptr{Float64}, g::Ptr{Float64},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int32}, n::Int32,
                                                    x::Ptr{Float128}, g::Ptr{Float128},
                                                    eval_f::Ptr{Cvoid}, eval_g::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

function tru_solve_without_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n,
                               x, g, eval_f, eval_g, eval_hprod, eval_prec)
  @ccall libgalahad_quadruple_64.tru_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                       userdata::Ptr{Cvoid},
                                                       status::Ptr{Int64}, n::Int64,
                                                       x::Ptr{Float128}, g::Ptr{Float128},
                                                       eval_f::Ptr{Cvoid},
                                                       eval_g::Ptr{Cvoid},
                                                       eval_hprod::Ptr{Cvoid},
                                                       eval_prec::Ptr{Cvoid})::Cvoid
end

export tru_solve_reverse_with_mat

function tru_solve_reverse_with_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float32}, f::Float32,
                                                      g::Ptr{Float32}, ne::Int32,
                                                      H_val::Ptr{Float32}, u::Ptr{Float32},
                                                      v::Ptr{Float32})::Cvoid
end

function tru_solve_reverse_with_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_single_64.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float32}, f::Float32,
                                                         g::Ptr{Float32}, ne::Int64,
                                                         H_val::Ptr{Float32},
                                                         u::Ptr{Float32},
                                                         v::Ptr{Float32})::Cvoid
end

function tru_solve_reverse_with_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32},
                                                      eval_status::Ptr{Int32}, n::Int32,
                                                      x::Ptr{Float64}, f::Float64,
                                                      g::Ptr{Float64}, ne::Int32,
                                                      H_val::Ptr{Float64}, u::Ptr{Float64},
                                                      v::Ptr{Float64})::Cvoid
end

function tru_solve_reverse_with_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_double_64.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         eval_status::Ptr{Int64}, n::Int64,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, ne::Int64,
                                                         H_val::Ptr{Float64},
                                                         u::Ptr{Float64},
                                                         v::Ptr{Float64})::Cvoid
end

function tru_solve_reverse_with_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float128}, f::Cfloat128,
                                                         g::Ptr{Float128}, ne::Int32,
                                                         H_val::Ptr{Float128},
                                                         u::Ptr{Float128},
                                                         v::Ptr{Float128})::Cvoid
end

function tru_solve_reverse_with_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                    eval_status, n, x, f, g, ne, H_val, u, v)
  @ccall libgalahad_quadruple_64.tru_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float128},
                                                            f::Cfloat128, g::Ptr{Float128},
                                                            ne::Int64, H_val::Ptr{Float128},
                                                            u::Ptr{Float128},
                                                            v::Ptr{Float128})::Cvoid
end

export tru_solve_reverse_without_mat

function tru_solve_reverse_without_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_single.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float32}, f::Float32,
                                                         g::Ptr{Float32}, u::Ptr{Float32},
                                                         v::Ptr{Float32})::Cvoid
end

function tru_solve_reverse_without_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_single_64.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float32},
                                                            f::Float32, g::Ptr{Float32},
                                                            u::Ptr{Float32},
                                                            v::Ptr{Float32})::Cvoid
end

function tru_solve_reverse_without_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_double.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32}, n::Int32,
                                                         x::Ptr{Float64}, f::Float64,
                                                         g::Ptr{Float64}, u::Ptr{Float64},
                                                         v::Ptr{Float64})::Cvoid
end

function tru_solve_reverse_without_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_double_64.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64},
                                                            eval_status::Ptr{Int64},
                                                            n::Int64, x::Ptr{Float64},
                                                            f::Float64, g::Ptr{Float64},
                                                            u::Ptr{Float64},
                                                            v::Ptr{Float64})::Cvoid
end

function tru_solve_reverse_without_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_quadruple.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int32},
                                                            eval_status::Ptr{Int32},
                                                            n::Int32, x::Ptr{Float128},
                                                            f::Cfloat128, g::Ptr{Float128},
                                                            u::Ptr{Float128},
                                                            v::Ptr{Float128})::Cvoid
end

function tru_solve_reverse_without_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                       eval_status, n, x, f, g, u, v)
  @ccall libgalahad_quadruple_64.tru_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                               status::Ptr{Int64},
                                                               eval_status::Ptr{Int64},
                                                               n::Int64, x::Ptr{Float128},
                                                               f::Cfloat128,
                                                               g::Ptr{Float128},
                                                               u::Ptr{Float128},
                                                               v::Ptr{Float128})::Cvoid
end

export tru_information

function tru_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.tru_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{tru_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function tru_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.tru_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{tru_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function tru_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.tru_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{tru_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function tru_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.tru_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{tru_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function tru_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.tru_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{tru_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function tru_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.tru_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{tru_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export tru_terminate

function tru_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{tru_control_type{Float32,Int32}},
                                         inform::Ptr{tru_inform_type{Float32,Int32}})::Cvoid
end

function tru_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{tru_control_type{Float32,Int64}},
                                            inform::Ptr{tru_inform_type{Float32,Int64}})::Cvoid
end

function tru_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{tru_control_type{Float64,Int32}},
                                         inform::Ptr{tru_inform_type{Float64,Int32}})::Cvoid
end

function tru_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{tru_control_type{Float64,Int64}},
                                            inform::Ptr{tru_inform_type{Float64,Int64}})::Cvoid
end

function tru_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{tru_control_type{Float128,Int32}},
                                            inform::Ptr{tru_inform_type{Float128,Int32}})::Cvoid
end

function tru_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.tru_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{tru_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{tru_inform_type{Float128,Int64}})::Cvoid
end
