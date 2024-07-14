export bnls_subproblem_control_type

struct bnls_subproblem_control_type{T}
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  alive_unit::Cint
  alive_file::NTuple{31,Cchar}
  jacobian_available::Cint
  hessian_available::Cint
  model::Cint
  norm::Cint
  non_monotone::Cint
  weight_update_strategy::Cint
  infinity::T
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
  rqs_control::rqs_control_type{T}
  glrt_control::glrt_control_type{T}
  psls_control::psls_control_type{T}
  bsc_control::bsc_control_type
  roots_control::roots_control_type{T}
end

export bnls_control_type

struct bnls_control_type{T}
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
  jacobian_available::Cint
  hessian_available::Cint
  model::Cint
  norm::Cint
  non_monotone::Cint
  weight_update_strategy::Cint
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
  rqs_control::rqs_control_type{T}
  glrt_control::glrt_control_type{T}
  psls_control::psls_control_type{T}
  bsc_control::bsc_control_type
  roots_control::roots_control_type{T}
  subproblem_control::bnls_subproblem_control_type{T}
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

struct bnls_subproblem_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::Cint
  cg_iter::Cint
  c_eval::Cint
  j_eval::Cint
  h_eval::Cint
  factorization_max::Cint
  factorization_status::Cint
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_c::T
  norm_g::T
  weight::T
  time::bnls_time_type{T}
  rqs_inform::rqs_inform_type{T}
  glrt_inform::glrt_inform_type{T}
  psls_inform::psls_inform_type{T}
  bsc_inform::bsc_inform_type{T}
  roots_inform::roots_inform_type
end

export bnls_inform_type

struct bnls_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::Cint
  cg_iter::Cint
  c_eval::Cint
  j_eval::Cint
  h_eval::Cint
  factorization_max::Cint
  factorization_status::Cint
  max_entries_factors::Int64
  factorization_integer::Int64
  factorization_real::Int64
  factorization_average::T
  obj::T
  norm_c::T
  norm_g::T
  weight::T
  time::bnls_time_type{T}
  rqs_inform::rqs_inform_type{T}
  glrt_inform::glrt_inform_type{T}
  psls_inform::psls_inform_type{T}
  bsc_inform::bsc_inform_type{T}
  roots_inform::roots_inform_type
  subproblem_inform::bnls_subproblem_inform_type{T}
end

export bnls_initialize_s

function bnls_initialize_s(data, control, inform)
  @ccall libgalahad_single.bnls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bnls_control_type{Float32}},
                                             inform::Ptr{bnls_inform_type{Float32}})::Cvoid
end

export bnls_initialize

function bnls_initialize(data, control, inform)
  @ccall libgalahad_double.bnls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bnls_control_type{Float64}},
                                           inform::Ptr{bnls_inform_type{Float64}})::Cvoid
end

export bnls_read_specfile_s

function bnls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.bnls_read_specfile_s(control::Ptr{bnls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export bnls_read_specfile

function bnls_read_specfile(control, specfile)
  @ccall libgalahad_double.bnls_read_specfile(control::Ptr{bnls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export bnls_import_s

function bnls_import_s(control, data, status, n, m, x_l, x_u, J_type, J_ne, J_row, J_col,
                       J_ptr, H_type, H_ne, H_row, H_col, H_ptr, P_type, P_ne, P_row, P_col,
                       P_ptr, w)
  @ccall libgalahad_single.bnls_import_s(control::Ptr{bnls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                         J_type::Ptr{Cchar}, J_ne::Cint, J_row::Ptr{Cint},
                                         J_col::Ptr{Cint}, J_ptr::Ptr{Cint},
                                         H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                         H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                         P_type::Ptr{Cchar}, P_ne::Cint, P_row::Ptr{Cint},
                                         P_col::Ptr{Cint}, P_ptr::Ptr{Cint},
                                         w::Ptr{Float32})::Cvoid
end

export bnls_import

function bnls_import(control, data, status, n, m, x_l, x_u, J_type, J_ne, J_row, J_col,
                     J_ptr, H_type, H_ne, H_row, H_col, H_ptr, P_type, P_ne, P_row, P_col,
                     P_ptr, w)
  @ccall libgalahad_double.bnls_import(control::Ptr{bnls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       m::Cint, x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                       J_type::Ptr{Cchar}, J_ne::Cint, J_row::Ptr{Cint},
                                       J_col::Ptr{Cint}, J_ptr::Ptr{Cint},
                                       H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                       H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                       P_type::Ptr{Cchar}, P_ne::Cint, P_row::Ptr{Cint},
                                       P_col::Ptr{Cint}, P_ptr::Ptr{Cint},
                                       w::Ptr{Float64})::Cvoid
end

export bnls_reset_control_s

function bnls_reset_control_s(control, data, status)
  @ccall libgalahad_single.bnls_reset_control_s(control::Ptr{bnls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export bnls_reset_control

function bnls_reset_control(control, data, status)
  @ccall libgalahad_double.bnls_reset_control(control::Ptr{bnls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export bnls_solve_with_mat_s

function bnls_solve_with_mat_s(data, userdata, status, n, m, x, c, g, eval_c, j_ne, eval_j,
                               h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_single.bnls_solve_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                 n::Cint, m::Cint, x::Ptr{Float32},
                                                 c::Ptr{Float32}, g::Ptr{Float32},
                                                 eval_c::Ptr{Cvoid}, j_ne::Cint,
                                                 eval_j::Ptr{Cvoid}, h_ne::Cint,
                                                 eval_h::Ptr{Cvoid}, p_ne::Cint,
                                                 eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_with_mat

function bnls_solve_with_mat(data, userdata, status, n, m, x, c, g, eval_c, j_ne, eval_j,
                             h_ne, eval_h, p_ne, eval_hprods)
  @ccall libgalahad_double.bnls_solve_with_mat(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                               status::Ptr{Cint}, n::Cint, m::Cint,
                                               x::Ptr{Float64}, c::Ptr{Float64},
                                               g::Ptr{Float64}, eval_c::Ptr{Cvoid},
                                               j_ne::Cint, eval_j::Ptr{Cvoid}, h_ne::Cint,
                                               eval_h::Ptr{Cvoid}, p_ne::Cint,
                                               eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_without_mat_s

function bnls_solve_without_mat_s(data, userdata, status, n, m, x, c, g, eval_c, eval_jprod,
                                  eval_hprod, p_ne, eval_hprods)
  @ccall libgalahad_single.bnls_solve_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                    n::Cint, m::Cint, x::Ptr{Float32},
                                                    c::Ptr{Float32}, g::Ptr{Float32},
                                                    eval_c::Ptr{Cvoid},
                                                    eval_jprod::Ptr{Cvoid},
                                                    eval_hprod::Ptr{Cvoid}, p_ne::Cint,
                                                    eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_without_mat

function bnls_solve_without_mat(data, userdata, status, n, m, x, c, g, eval_c, eval_jprod,
                                eval_hprod, p_ne, eval_hprods)
  @ccall libgalahad_double.bnls_solve_without_mat(data::Ptr{Ptr{Cvoid}},
                                                  userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                  n::Cint, m::Cint, x::Ptr{Float64},
                                                  c::Ptr{Float64}, g::Ptr{Float64},
                                                  eval_c::Ptr{Cvoid},
                                                  eval_jprod::Ptr{Cvoid},
                                                  eval_hprod::Ptr{Cvoid}, p_ne::Cint,
                                                  eval_hprods::Ptr{Cvoid})::Cvoid
end

export bnls_solve_reverse_with_mat_s

function bnls_solve_reverse_with_mat_s(data, status, eval_status, n, m, x, c, g, j_ne,
                                       J_val, y, h_ne, H_val, v, p_ne, P_val)
  @ccall libgalahad_single.bnls_solve_reverse_with_mat_s(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Cint},
                                                         eval_status::Ptr{Cint}, n::Cint,
                                                         m::Cint, x::Ptr{Float32},
                                                         c::Ptr{Float32}, g::Ptr{Float32},
                                                         j_ne::Cint, J_val::Ptr{Float32},
                                                         y::Ptr{Float32}, h_ne::Cint,
                                                         H_val::Ptr{Float32},
                                                         v::Ptr{Float32}, p_ne::Cint,
                                                         P_val::Ptr{Float32})::Cvoid
end

export bnls_solve_reverse_with_mat

function bnls_solve_reverse_with_mat(data, status, eval_status, n, m, x, c, g, j_ne, J_val,
                                     y, h_ne, H_val, v, p_ne, P_val)
  @ccall libgalahad_double.bnls_solve_reverse_with_mat(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint},
                                                       eval_status::Ptr{Cint}, n::Cint,
                                                       m::Cint, x::Ptr{Float64},
                                                       c::Ptr{Float64}, g::Ptr{Float64},
                                                       j_ne::Cint, J_val::Ptr{Float64},
                                                       y::Ptr{Float64}, h_ne::Cint,
                                                       H_val::Ptr{Float64}, v::Ptr{Float64},
                                                       p_ne::Cint,
                                                       P_val::Ptr{Float64})::Cvoid
end

export bnls_solve_reverse_without_mat_s

function bnls_solve_reverse_without_mat_s(data, status, eval_status, n, m, x, c, g,
                                          transpose, u, v, y, p_ne, P_val)
  @ccall libgalahad_single.bnls_solve_reverse_without_mat_s(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Cint},
                                                            eval_status::Ptr{Cint}, n::Cint,
                                                            m::Cint, x::Ptr{Float32},
                                                            c::Ptr{Float32},
                                                            g::Ptr{Float32},
                                                            transpose::Ptr{Bool},
                                                            u::Ptr{Float32},
                                                            v::Ptr{Float32},
                                                            y::Ptr{Float32}, p_ne::Cint,
                                                            P_val::Ptr{Float32})::Cvoid
end

export bnls_solve_reverse_without_mat

function bnls_solve_reverse_without_mat(data, status, eval_status, n, m, x, c, g, transpose,
                                        u, v, y, p_ne, P_val)
  @ccall libgalahad_double.bnls_solve_reverse_without_mat(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Cint},
                                                          eval_status::Ptr{Cint}, n::Cint,
                                                          m::Cint, x::Ptr{Float64},
                                                          c::Ptr{Float64}, g::Ptr{Float64},
                                                          transpose::Ptr{Bool},
                                                          u::Ptr{Float64}, v::Ptr{Float64},
                                                          y::Ptr{Float64}, p_ne::Cint,
                                                          P_val::Ptr{Float64})::Cvoid
end

export bnls_information_s

function bnls_information_s(data, inform, status)
  @ccall libgalahad_single.bnls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bnls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export bnls_information

function bnls_information(data, inform, status)
  @ccall libgalahad_double.bnls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{bnls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export bnls_terminate_s

function bnls_terminate_s(data, control, inform)
  @ccall libgalahad_single.bnls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bnls_control_type{Float32}},
                                            inform::Ptr{bnls_inform_type{Float32}})::Cvoid
end

export bnls_terminate

function bnls_terminate(data, control, inform)
  @ccall libgalahad_double.bnls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bnls_control_type{Float64}},
                                          inform::Ptr{bnls_inform_type{Float64}})::Cvoid
end
