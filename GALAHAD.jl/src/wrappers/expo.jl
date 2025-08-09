export expo_control_type

struct expo_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  max_it::INT
  max_eval::INT
  alive_unit::INT
  alive_file::NTuple{31,Cchar}
  update_multipliers_itmin::INT
  update_multipliers_tol::T
  infinity::T
  stop_abs_p::T
  stop_rel_p::T
  stop_abs_d::T
  stop_rel_d::T
  stop_abs_c::T
  stop_rel_c::T
  stop_s::T
  stop_subproblem_rel::T
  initial_mu::T
  mu_reduce::T
  obj_unbounded::T
  try_advanced_start::T
  try_sqp_start::T
  stop_advanced_start::T
  cpu_time_limit::T
  clock_time_limit::T
  hessian_available::Bool
  subproblem_direct::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  bsc_control::bsc_control_type{INT}
  tru_control::tru_control_type{T,INT}
  ssls_control::ssls_control_type{T,INT}
end

export expo_time_type

struct expo_time_type{T}
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

export expo_inform_type

struct expo_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::INT
  fc_eval::INT
  gj_eval::INT
  hl_eval::INT
  obj::T
  primal_infeasibility::T
  dual_infeasibility::T
  complementary_slackness::T
  time::expo_time_type{T}
  bsc_inform::bsc_inform_type{T,INT}
  tru_inform::tru_inform_type{T,INT}
  ssls_inform::ssls_inform_type{T,INT}
end

export expo_initialize

function expo_initialize(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{expo_control_type{Float32,Int32}},
                                           inform::Ptr{expo_inform_type{Float32,Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function expo_initialize(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{expo_control_type{Float32,Int64}},
                                              inform::Ptr{expo_inform_type{Float32,Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function expo_initialize(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{expo_control_type{Float64,Int32}},
                                           inform::Ptr{expo_inform_type{Float64,Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function expo_initialize(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{expo_control_type{Float64,Int64}},
                                              inform::Ptr{expo_inform_type{Float64,Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function expo_initialize(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{expo_control_type{Float128,
                                                                             Int32}},
                                              inform::Ptr{expo_inform_type{Float128,Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function expo_initialize(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.expo_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{expo_control_type{Float128,
                                                                                Int64}},
                                                 inform::Ptr{expo_inform_type{Float128,
                                                                              Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export expo_read_specfile

function expo_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.expo_read_specfile(control::Ptr{expo_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function expo_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.expo_read_specfile(control::Ptr{expo_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function expo_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.expo_read_specfile(control::Ptr{expo_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function expo_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.expo_read_specfile(control::Ptr{expo_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function expo_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.expo_read_specfile(control::Ptr{expo_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function expo_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.expo_read_specfile(control::Ptr{expo_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export expo_import

function expo_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.expo_import(control::Ptr{expo_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, J_type::Ptr{Cchar}, J_ne::Int32,
                                       J_row::Ptr{Int32}, J_col::Ptr{Int32},
                                       J_ptr::Ptr{Int32}, H_type::Ptr{Cchar}, H_ne::Int32,
                                       H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                       H_ptr::Ptr{Int32})::Cvoid
end

function expo_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.expo_import(control::Ptr{expo_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, J_type::Ptr{Cchar},
                                          J_ne::Int64, J_row::Ptr{Int64}, J_col::Ptr{Int64},
                                          J_ptr::Ptr{Int64}, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64})::Cvoid
end

function expo_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.expo_import(control::Ptr{expo_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, J_type::Ptr{Cchar}, J_ne::Int32,
                                       J_row::Ptr{Int32}, J_col::Ptr{Int32},
                                       J_ptr::Ptr{Int32}, H_type::Ptr{Cchar}, H_ne::Int32,
                                       H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                       H_ptr::Ptr{Int32})::Cvoid
end

function expo_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.expo_import(control::Ptr{expo_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, J_type::Ptr{Cchar},
                                          J_ne::Int64, J_row::Ptr{Int64}, J_col::Ptr{Int64},
                                          J_ptr::Ptr{Int64}, H_type::Ptr{Cchar},
                                          H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                          H_ptr::Ptr{Int64})::Cvoid
end

function expo_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.expo_import(control::Ptr{expo_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, J_type::Ptr{Cchar},
                                          J_ne::Int32, J_row::Ptr{Int32}, J_col::Ptr{Int32},
                                          J_ptr::Ptr{Int32}, H_type::Ptr{Cchar},
                                          H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                          H_ptr::Ptr{Int32})::Cvoid
end

function expo_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, J_type,
                     J_ne, J_row, J_col, J_ptr, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.expo_import(control::Ptr{expo_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, J_type::Ptr{Cchar},
                                             J_ne::Int64, J_row::Ptr{Int64},
                                             J_col::Ptr{Int64}, J_ptr::Ptr{Int64},
                                             H_type::Ptr{Cchar}, H_ne::Int64,
                                             H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                             H_ptr::Ptr{Int64})::Cvoid
end

export expo_reset_control

function expo_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.expo_reset_control(control::Ptr{expo_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function expo_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.expo_reset_control(control::Ptr{expo_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function expo_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.expo_reset_control(control::Ptr{expo_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function expo_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.expo_reset_control(control::Ptr{expo_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function expo_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.expo_reset_control(control::Ptr{expo_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function expo_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.expo_reset_control(control::Ptr{expo_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export expo_solve_hessian_direct

function expo_solve_hessian_direct(::Type{Float32}, ::Type{Int32}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_single.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int32}, n::Int32, m::Int32,
                                                     J_ne::Int32, H_ne::Int32,
                                                     c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                                     x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                     x::Ptr{Float32}, y::Ptr{Float32},
                                                     z::Ptr{Float32}, c::Ptr{Float32},
                                                     gl::Ptr{Float32}, eval_fc::Ptr{Cvoid},
                                                     eval_gj::Ptr{Cvoid},
                                                     eval_hl::Ptr{Cvoid})::Cvoid
end

function expo_solve_hessian_direct(::Type{Float32}, ::Type{Int64}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_single_64.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                        userdata::Ptr{Cvoid},
                                                        status::Ptr{Int64}, n::Int64,
                                                        m::Int64, J_ne::Int64, H_ne::Int64,
                                                        c_l::Ptr{Float32},
                                                        c_u::Ptr{Float32},
                                                        x_l::Ptr{Float32},
                                                        x_u::Ptr{Float32}, x::Ptr{Float32},
                                                        y::Ptr{Float32}, z::Ptr{Float32},
                                                        c::Ptr{Float32}, gl::Ptr{Float32},
                                                        eval_fc::Ptr{Cvoid},
                                                        eval_gj::Ptr{Cvoid},
                                                        eval_hl::Ptr{Cvoid})::Cvoid
end

function expo_solve_hessian_direct(::Type{Float64}, ::Type{Int32}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_double.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int32}, n::Int32, m::Int32,
                                                     J_ne::Int32, H_ne::Int32,
                                                     c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                     x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                     x::Ptr{Float64}, y::Ptr{Float64},
                                                     z::Ptr{Float64}, c::Ptr{Float64},
                                                     gl::Ptr{Float64}, eval_fc::Ptr{Cvoid},
                                                     eval_gj::Ptr{Cvoid},
                                                     eval_hl::Ptr{Cvoid})::Cvoid
end

function expo_solve_hessian_direct(::Type{Float64}, ::Type{Int64}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_double_64.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                        userdata::Ptr{Cvoid},
                                                        status::Ptr{Int64}, n::Int64,
                                                        m::Int64, J_ne::Int64, H_ne::Int64,
                                                        c_l::Ptr{Float64},
                                                        c_u::Ptr{Float64},
                                                        x_l::Ptr{Float64},
                                                        x_u::Ptr{Float64}, x::Ptr{Float64},
                                                        y::Ptr{Float64}, z::Ptr{Float64},
                                                        c::Ptr{Float64}, gl::Ptr{Float64},
                                                        eval_fc::Ptr{Cvoid},
                                                        eval_gj::Ptr{Cvoid},
                                                        eval_hl::Ptr{Cvoid})::Cvoid
end

function expo_solve_hessian_direct(::Type{Float128}, ::Type{Int32}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_quadruple.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                        userdata::Ptr{Cvoid},
                                                        status::Ptr{Int32}, n::Int32,
                                                        m::Int32, J_ne::Int32, H_ne::Int32,
                                                        c_l::Ptr{Float128},
                                                        c_u::Ptr{Float128},
                                                        x_l::Ptr{Float128},
                                                        x_u::Ptr{Float128},
                                                        x::Ptr{Float128}, y::Ptr{Float128},
                                                        z::Ptr{Float128}, c::Ptr{Float128},
                                                        gl::Ptr{Float128},
                                                        eval_fc::Ptr{Cvoid},
                                                        eval_gj::Ptr{Cvoid},
                                                        eval_hl::Ptr{Cvoid})::Cvoid
end

function expo_solve_hessian_direct(::Type{Float128}, ::Type{Int64}, data, userdata, status,
                                   n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x, y, z, c, gl,
                                   eval_fc, eval_gj, eval_hl)
  @ccall libgalahad_quadruple_64.expo_solve_hessian_direct(data::Ptr{Ptr{Cvoid}},
                                                           userdata::Ptr{Cvoid},
                                                           status::Ptr{Int64}, n::Int64,
                                                           m::Int64, J_ne::Int64,
                                                           H_ne::Int64, c_l::Ptr{Float128},
                                                           c_u::Ptr{Float128},
                                                           x_l::Ptr{Float128},
                                                           x_u::Ptr{Float128},
                                                           x::Ptr{Float128},
                                                           y::Ptr{Float128},
                                                           z::Ptr{Float128},
                                                           c::Ptr{Float128},
                                                           gl::Ptr{Float128},
                                                           eval_fc::Ptr{Cvoid},
                                                           eval_gj::Ptr{Cvoid},
                                                           eval_hl::Ptr{Cvoid})::Cvoid
end

export expo_information

function expo_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.expo_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{expo_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function expo_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.expo_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{expo_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function expo_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.expo_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{expo_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function expo_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.expo_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{expo_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function expo_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.expo_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{expo_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function expo_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.expo_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{expo_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export expo_terminate

function expo_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{expo_control_type{Float32,Int32}},
                                          inform::Ptr{expo_inform_type{Float32,Int32}})::Cvoid
end

function expo_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{expo_control_type{Float32,Int64}},
                                             inform::Ptr{expo_inform_type{Float32,Int64}})::Cvoid
end

function expo_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{expo_control_type{Float64,Int32}},
                                          inform::Ptr{expo_inform_type{Float64,Int32}})::Cvoid
end

function expo_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{expo_control_type{Float64,Int64}},
                                             inform::Ptr{expo_inform_type{Float64,Int64}})::Cvoid
end

function expo_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{expo_control_type{Float128,Int32}},
                                             inform::Ptr{expo_inform_type{Float128,Int32}})::Cvoid
end

function expo_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.expo_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{expo_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{expo_inform_type{Float128,
                                                                             Int64}})::Cvoid
end

function run_sif(::Val{:expo}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runexpo_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:expo}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runexpo_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
