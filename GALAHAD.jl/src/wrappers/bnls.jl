export bnls_subproblem_control_type

struct bnls_subproblem_control_type{T,INT}
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  alive_unit::INT
  alive_file::NTuple{31,Cchar}
  jacobian_available::INT
  hessian_available::INT
  model::INT
  norm::INT
  non_monotone::INT
  weight_update_strategy::INT
  stop_c_absolute::T
  stop_c_relative::T
  stop_g_absolute::T
  stop_g_relative::T
  stop_s::T
  power::T
  initial_weight::T
  minimum_weight::T
  initial_inner_weight::T
  eta_successful::T
  eta_very_successful::T
  eta_too_successful::T
  weight_decrease_min::T
  weight_decrease::T
  weight_increase::T
  weight_increase_max::T
  reduce_gap::T
  tiny_gap::T
  large_root::T
  switch_to_newton::T
  cpu_time_limit::T
  clock_time_limit::T
  subproblem_direct::Bool
  renormalize_weight::Bool
  magic_step::Bool
  print_obj::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  rqs_control::rqs_control_type{T,INT}
  glrt_control::glrt_control_type{T,INT}
  psls_control::psls_control_type{T,INT}
  bsc_control::bsc_control_type{INT}
  roots_control::roots_control_type{T,INT}
end

export bnls_control_type

struct bnls_control_type{T,INT}
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
  jacobian_available::INT
  hessian_available::INT
  model::INT
  norm::INT
  non_monotone::INT
  weight_update_strategy::INT
  stop_c_absolute::T
  stop_c_relative::T
  stop_g_absolute::T
  stop_g_relative::T
  stop_s::T
  power::T
  initial_weight::T
  minimum_weight::T
  initial_inner_weight::T
  eta_successful::T
  eta_very_successful::T
  eta_too_successful::T
  weight_decrease_min::T
  weight_decrease::T
  weight_increase::T
  weight_increase_max::T
  reduce_gap::T
  tiny_gap::T
  large_root::T
  switch_to_newton::T
  cpu_time_limit::T
  clock_time_limit::T
  subproblem_direct::Bool
  renormalize_weight::Bool
  magic_step::Bool
  print_obj::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  rqs_control::rqs_control_type{T,INT}
  glrt_control::glrt_control_type{T,INT}
  psls_control::psls_control_type{T,INT}
  bsc_control::bsc_control_type{INT}
  roots_control::roots_control_type{T,INT}
  subproblem_control::bnls_subproblem_control_type{T,INT}
end

export bnls_time_type

struct bnls_time_type{T}
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

export bnls_subproblem_inform_type

struct bnls_subproblem_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::INT
  cg_iter::INT
  c_eval::INT
  j_eval::INT
  h_eval::INT
  factorization_max::INT
  factorization_status::INT
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_c::T
  norm_g::T
  weight::T
  time::bnls_time_type{T}
  rqs_inform::rqs_inform_type{T,INT}
  glrt_inform::glrt_inform_type{T,INT}
  psls_inform::psls_inform_type{T,INT}
  bsc_inform::bsc_inform_type{T,INT}
  roots_inform::roots_inform_type{INT}
end

export bnls_inform_type

struct bnls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::INT
  cg_iter::INT
  c_eval::INT
  j_eval::INT
  h_eval::INT
  factorization_max::INT
  factorization_status::INT
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_c::T
  norm_g::T
  weight::T
  time::bnls_time_type{T}
  rqs_inform::rqs_inform_type{T,INT}
  glrt_inform::glrt_inform_type{T,INT}
  psls_inform::psls_inform_type{T,INT}
  bsc_inform::bsc_inform_type{T,INT}
  roots_inform::roots_inform_type{INT}
  subproblem_inform::bnls_subproblem_inform_type{T,INT}
end

export bnls_initialize

function bnls_initialize(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bnls_control_type{Float32,Int32}},
                                           inform::Ptr{bnls_inform_type{Float32,Int32}})::Cvoid
end

function bnls_initialize(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{bnls_control_type{Float32,Int64}},
                                              inform::Ptr{bnls_inform_type{Float32,Int64}})::Cvoid
end

function bnls_initialize(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bnls_control_type{Float64,Int32}},
                                           inform::Ptr{bnls_inform_type{Float64,Int32}})::Cvoid
end

function bnls_initialize(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{bnls_control_type{Float64,Int64}},
                                              inform::Ptr{bnls_inform_type{Float64,Int64}})::Cvoid
end

function bnls_initialize(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{bnls_control_type{Float128,
                                                                             Int32}},
                                              inform::Ptr{bnls_inform_type{Float128,Int32}})::Cvoid
end

function bnls_initialize(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{bnls_control_type{Float128,
                                                                                Int64}},
                                                 inform::Ptr{bnls_inform_type{Float128,
                                                                              Int64}})::Cvoid
end

export bnls_read_specfile

function bnls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.bnls_read_specfile(control::Ptr{bnls_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function bnls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.bnls_read_specfile(control::Ptr{bnls_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function bnls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.bnls_read_specfile(control::Ptr{bnls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function bnls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.bnls_read_specfile(control::Ptr{bnls_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function bnls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.bnls_read_specfile(control::Ptr{bnls_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function bnls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.bnls_read_specfile(control::Ptr{bnls_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export bnls_import

function bnls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_single.bnls_import(control::Ptr{bnls_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                       J_type::Ptr{Cchar}, J_ne::Int32, J_row::Ptr{Int32},
                                       J_col::Ptr{Int32}, J_ptr::Ptr{Int32},
                                       H_type::Ptr{Cchar}, H_ne::Int32, H_row::Ptr{Int32},
                                       H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                       P_type::Ptr{Cchar}, P_ne::Int32, P_row::Ptr{Int32},
                                       P_col::Ptr{Int32}, P_ptr::Ptr{Int32},
                                       w::Ptr{Float32})::Cvoid
end

function bnls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_single_64.bnls_import(control::Ptr{bnls_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32}, J_type::Ptr{Cchar},
                                          J_ne::Int64, J_row::Ptr{Int64}, J_col::Ptr{Int64},
                                          J_ptr::Ptr{Int64}, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64}, P_type::Ptr{Cchar},
                                          P_ne::Int64, P_row::Ptr{Int64}, P_col::Ptr{Int64},
                                          P_ptr::Ptr{Int64}, w::Ptr{Float32})::Cvoid
end

function bnls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_double.bnls_import(control::Ptr{bnls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                       J_type::Ptr{Cchar}, J_ne::Int32, J_row::Ptr{Int32},
                                       J_col::Ptr{Int32}, J_ptr::Ptr{Int32},
                                       H_type::Ptr{Cchar}, H_ne::Int32, H_row::Ptr{Int32},
                                       H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                       P_type::Ptr{Cchar}, P_ne::Int32, P_row::Ptr{Int32},
                                       P_col::Ptr{Int32}, P_ptr::Ptr{Int32},
                                       w::Ptr{Float64})::Cvoid
end

function bnls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_double_64.bnls_import(control::Ptr{bnls_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, J_type::Ptr{Cchar},
                                          J_ne::Int64, J_row::Ptr{Int64}, J_col::Ptr{Int64},
                                          J_ptr::Ptr{Int64}, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64}, P_type::Ptr{Cchar},
                                          P_ne::Int64, P_row::Ptr{Int64}, P_col::Ptr{Int64},
                                          P_ptr::Ptr{Int64}, w::Ptr{Float64})::Cvoid
end

function bnls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_quadruple.bnls_import(control::Ptr{bnls_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, x_l::Ptr{Float128},
                                          x_u::Ptr{Float128}, J_type::Ptr{Cchar},
                                          J_ne::Int32, J_row::Ptr{Int32}, J_col::Ptr{Int32},
                                          J_ptr::Ptr{Int32}, H_type::Ptr{Cchar},
                                          H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                          H_ptr::Ptr{Int32}, P_type::Ptr{Cchar},
                                          P_ne::Int32, P_row::Ptr{Int32}, P_col::Ptr{Int32},
                                          P_ptr::Ptr{Int32}, w::Ptr{Float128})::Cvoid
end

function bnls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, x_l, x_u,
                     J_type, J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr,
                     P_type, P_ne, P_row, P_col, P_ptr, w)
  @ccall libgalahad_quadruple_64.bnls_import(control::Ptr{bnls_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, x_l::Ptr{Float128},
                                             x_u::Ptr{Float128}, J_type::Ptr{Cchar},
                                             J_ne::Int64, J_row::Ptr{Int64},
                                             J_col::Ptr{Int64}, J_ptr::Ptr{Int64},
                                             H_type::Ptr{Cchar}, H_ne::Int64,
                                             H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                             H_ptr::Ptr{Int64}, P_type::Ptr{Cchar},
                                             P_ne::Int64, P_row::Ptr{Int64},
                                             P_col::Ptr{Int64}, P_ptr::Ptr{Int64},
                                             w::Ptr{Float128})::Cvoid
end

export bnls_reset_control

function bnls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.bnls_reset_control(control::Ptr{bnls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function bnls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.bnls_reset_control(control::Ptr{bnls_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function bnls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.bnls_reset_control(control::Ptr{bnls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function bnls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.bnls_reset_control(control::Ptr{bnls_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function bnls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.bnls_reset_control(control::Ptr{bnls_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function bnls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.bnls_reset_control(control::Ptr{bnls_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export bnls_solve_with_mat

function bnls_solve_with_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_single.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Int32}, n::Int32, m::Int32,
                                               x::Ptr{Float32}, c::Ptr{Float32},
                                               g::Ptr{Float32}, eval_c::Ptr{Cvoid},
                                               j_ne::Int32, eval_j::Ptr{Cvoid}, h_ne::Int32,
                                               eval_h::Ptr{Cvoid}, p_ne::Int32,
                                               eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_with_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_single_64.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                  n::Int64, m::Int64, x::Ptr{Float32},
                                                  c::Ptr{Float32}, g::Ptr{Float32},
                                                  eval_c::Ptr{Cvoid}, j_ne::Int64,
                                                  eval_j::Ptr{Cvoid}, h_ne::Int64,
                                                  eval_h::Ptr{Cvoid}, p_ne::Int64,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_with_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_double.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Int32}, n::Int32, m::Int32,
                                               x::Ptr{Float64}, c::Ptr{Float64},
                                               g::Ptr{Float64}, eval_c::Ptr{Cvoid},
                                               j_ne::Int32, eval_j::Ptr{Cvoid}, h_ne::Int32,
                                               eval_h::Ptr{Cvoid}, p_ne::Int32,
                                               eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_with_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_double_64.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                  n::Int64, m::Int64, x::Ptr{Float64},
                                                  c::Ptr{Float64}, g::Ptr{Float64},
                                                  eval_c::Ptr{Cvoid}, j_ne::Int64,
                                                  eval_j::Ptr{Cvoid}, h_ne::Int64,
                                                  eval_h::Ptr{Cvoid}, p_ne::Int64,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_with_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_quadruple.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                  n::Int32, m::Int32, x::Ptr{Float128},
                                                  c::Ptr{Float128}, g::Ptr{Float128},
                                                  eval_c::Ptr{Cvoid}, j_ne::Int32,
                                                  eval_j::Ptr{Cvoid}, h_ne::Int32,
                                                  eval_h::Ptr{Cvoid}, p_ne::Int32,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_with_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, m,
                             x, c, g, eval_c, j_ne, eval_j, h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_quadruple_64.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int64}, n::Int64, m::Int64,
                                                     x::Ptr{Float128}, c::Ptr{Float128},
                                                     g::Ptr{Float128}, eval_c::Ptr{Cvoid},
                                                     j_ne::Int64, eval_j::Ptr{Cvoid},
                                                     h_ne::Int64, eval_h::Ptr{Cvoid},
                                                     p_ne::Int64,
                                                     eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_without_mat

function bnls_solve_without_mat(::Type{Float32}, ::Type{Int32}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_single.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                  n::Int32, m::Int32, x::Ptr{Float32},
                                                  c::Ptr{Float32}, g::Ptr{Float32},
                                                  eval_c::Ptr{Cvoid},
                                                  eval_jprod::Ptr{Cvoid},
                                                  eval_hprod::Ptr{Cvoid}, p_ne::Int32,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_without_mat(::Type{Float32}, ::Type{Int64}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_single_64.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int64}, n::Int64, m::Int64,
                                                     x::Ptr{Float32}, c::Ptr{Float32},
                                                     g::Ptr{Float32}, eval_c::Ptr{Cvoid},
                                                     eval_jprod::Ptr{Cvoid},
                                                     eval_hprod::Ptr{Cvoid}, p_ne::Int64,
                                                     eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_without_mat(::Type{Float64}, ::Type{Int32}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_double.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                  n::Int32, m::Int32, x::Ptr{Float64},
                                                  c::Ptr{Float64}, g::Ptr{Float64},
                                                  eval_c::Ptr{Cvoid},
                                                  eval_jprod::Ptr{Cvoid},
                                                  eval_hprod::Ptr{Cvoid}, p_ne::Int32,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_without_mat(::Type{Float64}, ::Type{Int64}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_double_64.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int64}, n::Int64, m::Int64,
                                                     x::Ptr{Float64}, c::Ptr{Float64},
                                                     g::Ptr{Float64}, eval_c::Ptr{Cvoid},
                                                     eval_jprod::Ptr{Cvoid},
                                                     eval_hprod::Ptr{Cvoid}, p_ne::Int64,
                                                     eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_without_mat(::Type{Float128}, ::Type{Int32}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_quadruple.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int32}, n::Int32, m::Int32,
                                                     x::Ptr{Float128}, c::Ptr{Float128},
                                                     g::Ptr{Float128}, eval_c::Ptr{Cvoid},
                                                     eval_jprod::Ptr{Cvoid},
                                                     eval_hprod::Ptr{Cvoid}, p_ne::Int32,
                                                     eval_hprods::Ptr{Cvoid})::Cvoid
end

function bnls_solve_without_mat(::Type{Float128}, ::Type{Int64}, data, userdata, status, n,
                                m, x, c, g, eval_c, eval_jprod, eval_hprod, p_ne,
                                eval_hprods)
  @ccall libgalahad_quadruple_64.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                        userdata::Ptr{Cvoid},
                                                        status::Ptr{Int64}, n::Int64,
                                                        m::Int64, x::Ptr{Float128},
                                                        c::Ptr{Float128}, g::Ptr{Float128},
                                                        eval_c::Ptr{Cvoid},
                                                        eval_jprod::Ptr{Cvoid},
                                                        eval_hprod::Ptr{Cvoid}, p_ne::Int64,
                                                        eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_reverse_with_mat

function bnls_solve_reverse_with_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_single.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32},
                                                       eval_status::Ptr{Int32}, n::Int32,
                                                       m::Int32, x::Ptr{Float32},
                                                       c::Ptr{Float32}, g::Ptr{Float32},
                                                       j_ne::Int32, J_val::Ptr{Float32},
                                                       y::Ptr{Float32}, h_ne::Int32,
                                                       H_val::Ptr{Float32}, v::Ptr{Float32},
                                                       p_ne::Int32,
                                                       P_val::Ptr{Float32})::Cvoid
end

function bnls_solve_reverse_with_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_single_64.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64},
                                                          eval_status::Ptr{Int64}, n::Int64,
                                                          m::Int64, x::Ptr{Float32},
                                                          c::Ptr{Float32}, g::Ptr{Float32},
                                                          j_ne::Int64, J_val::Ptr{Float32},
                                                          y::Ptr{Float32}, h_ne::Int64,
                                                          H_val::Ptr{Float32},
                                                          v::Ptr{Float32}, p_ne::Int64,
                                                          P_val::Ptr{Float32})::Cvoid
end

function bnls_solve_reverse_with_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_double.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32},
                                                       eval_status::Ptr{Int32}, n::Int32,
                                                       m::Int32, x::Ptr{Float64},
                                                       c::Ptr{Float64}, g::Ptr{Float64},
                                                       j_ne::Int32, J_val::Ptr{Float64},
                                                       y::Ptr{Float64}, h_ne::Int32,
                                                       H_val::Ptr{Float64}, v::Ptr{Float64},
                                                       p_ne::Int32,
                                                       P_val::Ptr{Float64})::Cvoid
end

function bnls_solve_reverse_with_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_double_64.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64},
                                                          eval_status::Ptr{Int64}, n::Int64,
                                                          m::Int64, x::Ptr{Float64},
                                                          c::Ptr{Float64}, g::Ptr{Float64},
                                                          j_ne::Int64, J_val::Ptr{Float64},
                                                          y::Ptr{Float64}, h_ne::Int64,
                                                          H_val::Ptr{Float64},
                                                          v::Ptr{Float64}, p_ne::Int64,
                                                          P_val::Ptr{Float64})::Cvoid
end

function bnls_solve_reverse_with_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_quadruple.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int32},
                                                          eval_status::Ptr{Int32}, n::Int32,
                                                          m::Int32, x::Ptr{Float128},
                                                          c::Ptr{Float128},
                                                          g::Ptr{Float128}, j_ne::Int32,
                                                          J_val::Ptr{Float128},
                                                          y::Ptr{Float128}, h_ne::Int32,
                                                          H_val::Ptr{Float128},
                                                          v::Ptr{Float128}, p_ne::Int32,
                                                          P_val::Ptr{Float128})::Cvoid
end

function bnls_solve_reverse_with_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                     eval_status, n, m, x, c, g, j_ne, J_val, y, h_ne,
                                     H_val, v, p_ne, P_val)
  @ccall libgalahad_quadruple_64.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int64},
                                                             eval_status::Ptr{Int64},
                                                             n::Int64, m::Int64,
                                                             x::Ptr{Float128},
                                                             c::Ptr{Float128},
                                                             g::Ptr{Float128}, j_ne::Int64,
                                                             J_val::Ptr{Float128},
                                                             y::Ptr{Float128}, h_ne::Int64,
                                                             H_val::Ptr{Float128},
                                                             v::Ptr{Float128}, p_ne::Int64,
                                                             P_val::Ptr{Float128})::Cvoid
end

export bnls_solve_reverse_without_mat

function bnls_solve_reverse_without_mat(::Type{Float32}, ::Type{Int32}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_single.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int32},
                                                          eval_status::Ptr{Int32}, n::Int32,
                                                          m::Int32, x::Ptr{Float32},
                                                          c::Ptr{Float32}, g::Ptr{Float32},
                                                          transpose::Ptr{Bool},
                                                          u::Ptr{Float32}, v::Ptr{Float32},
                                                          y::Ptr{Float32}, p_ne::Int32,
                                                          P_val::Ptr{Float32})::Cvoid
end

function bnls_solve_reverse_without_mat(::Type{Float32}, ::Type{Int64}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_single_64.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int64},
                                                             eval_status::Ptr{Int64},
                                                             n::Int64, m::Int64,
                                                             x::Ptr{Float32},
                                                             c::Ptr{Float32},
                                                             g::Ptr{Float32},
                                                             transpose::Ptr{Bool},
                                                             u::Ptr{Float32},
                                                             v::Ptr{Float32},
                                                             y::Ptr{Float32}, p_ne::Int64,
                                                             P_val::Ptr{Float32})::Cvoid
end

function bnls_solve_reverse_without_mat(::Type{Float64}, ::Type{Int32}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_double.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int32},
                                                          eval_status::Ptr{Int32}, n::Int32,
                                                          m::Int32, x::Ptr{Float64},
                                                          c::Ptr{Float64}, g::Ptr{Float64},
                                                          transpose::Ptr{Bool},
                                                          u::Ptr{Float64}, v::Ptr{Float64},
                                                          y::Ptr{Float64}, p_ne::Int32,
                                                          P_val::Ptr{Float64})::Cvoid
end

function bnls_solve_reverse_without_mat(::Type{Float64}, ::Type{Int64}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_double_64.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int64},
                                                             eval_status::Ptr{Int64},
                                                             n::Int64, m::Int64,
                                                             x::Ptr{Float64},
                                                             c::Ptr{Float64},
                                                             g::Ptr{Float64},
                                                             transpose::Ptr{Bool},
                                                             u::Ptr{Float64},
                                                             v::Ptr{Float64},
                                                             y::Ptr{Float64}, p_ne::Int64,
                                                             P_val::Ptr{Float64})::Cvoid
end

function bnls_solve_reverse_without_mat(::Type{Float128}, ::Type{Int32}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_quadruple.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int32},
                                                             eval_status::Ptr{Int32},
                                                             n::Int32, m::Int32,
                                                             x::Ptr{Float128},
                                                             c::Ptr{Float128},
                                                             g::Ptr{Float128},
                                                             transpose::Ptr{Bool},
                                                             u::Ptr{Float128},
                                                             v::Ptr{Float128},
                                                             y::Ptr{Float128}, p_ne::Int32,
                                                             P_val::Ptr{Float128})::Cvoid
end

function bnls_solve_reverse_without_mat(::Type{Float128}, ::Type{Int64}, data, status,
                                        eval_status, n, m, x, c, g, transpose, u, v, y,
                                        p_ne, P_val)
  @ccall libgalahad_quadruple_64.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                                status::Ptr{Int64},
                                                                eval_status::Ptr{Int64},
                                                                n::Int64, m::Int64,
                                                                x::Ptr{Float128},
                                                                c::Ptr{Float128},
                                                                g::Ptr{Float128},
                                                                transpose::Ptr{Bool},
                                                                u::Ptr{Float128},
                                                                v::Ptr{Float128},
                                                                y::Ptr{Float128},
                                                                p_ne::Int64,
                                                                P_val::Ptr{Float128})::Cvoid
end

export bnls_information

function bnls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.bnls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{bnls_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function bnls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.bnls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{bnls_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function bnls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.bnls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{bnls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function bnls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.bnls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{bnls_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function bnls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.bnls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{bnls_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function bnls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.bnls_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{bnls_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export bnls_terminate

function bnls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bnls_control_type{Float32,Int32}},
                                          inform::Ptr{bnls_inform_type{Float32,Int32}})::Cvoid
end

function bnls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bnls_control_type{Float32,Int64}},
                                             inform::Ptr{bnls_inform_type{Float32,Int64}})::Cvoid
end

function bnls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bnls_control_type{Float64,Int32}},
                                          inform::Ptr{bnls_inform_type{Float64,Int32}})::Cvoid
end

function bnls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bnls_control_type{Float64,Int64}},
                                             inform::Ptr{bnls_inform_type{Float64,Int64}})::Cvoid
end

function bnls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bnls_control_type{Float128,Int32}},
                                             inform::Ptr{bnls_inform_type{Float128,Int32}})::Cvoid
end

function bnls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bnls_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{bnls_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
