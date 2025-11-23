export trek_control_type

struct trek_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  eks_max::INT
  it_max::INT
  f:T
  reduction:T
  stop_residual:T
  reorthogonalize::Bool
  s_version_52::Bool
  perturb_c::Bool
  stop_check_all_orders::Bool
  new_radius::Bool
  new_values::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  linear_solver::NTuple{31,Cchar}
  linear_solver_for_s::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
  sls_s_control::sls_control_type{T,INT}
  trs_control::trs_control_type{T,INT}
end

export trek_time_type

struct trek_time_type{T}
  total::T
  assemble::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_assemble::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export trek_inform_type

struct trek_inform_type{T,INT}
  status::INT
  alloc_status::INT
  iter::INT
  n_vec::INT
  obj::T
  x_norm::T
  multiplier::T
  radius::T
  next_radius::T
  error::T
  bad_alloc::NTuple{81,Cchar}
  time::trek_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  sls_s_inform::sls_inform_type{T,INT}
  trs_inform::trs_inform_type{T,INT}
end

export trek_initialize

function trek_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.trek_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trek_control_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function trek_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.trek_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{trek_control_type{Float32,
                                                                                Int64}},
                                                  status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function trek_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.trek_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{trek_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function trek_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.trek_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{trek_control_type{Float64,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function trek_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.trek_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{trek_control_type{Float128,
                                                                             Int32}},
                                               status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function trek_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.trek_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{trek_control_type{Float128,
                                                                                   Int64}},
                                                     status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export trek_read_specfile

function trek_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.trek_read_specfile_s(control::Ptr{trek_control_type{Float32,Int32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function trek_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.trek_read_specfile_s_64(control::Ptr{trek_control_type{Float32,
                                                                                   Int64}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function trek_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.trek_read_specfile(control::Ptr{trek_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function trek_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.trek_read_specfile_64(control::Ptr{trek_control_type{Float64,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function trek_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.trek_read_specfile_q(control::Ptr{trek_control_type{Float128,
                                                                                Int32}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

function trek_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.trek_read_specfile_q_64(control::Ptr{trek_control_type{Float128,
                                                                                      Int64}},
                                                        specfile::Ptr{Cchar})::Cvoid
end

export trek_import

function trek_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single.trek_s_import(control::Ptr{trek_control_type{Float32,Int32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        H_type::Ptr{Cchar}, H_ne::Int32, H_row::Ptr{Int32},
                                        H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function trek_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.trek_s_import_64(control::Ptr{trek_control_type{Float32,Int64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                              H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                              H_ptr::Ptr{Int64})::Cvoid
end

function trek_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double.trek_import(control::Ptr{trek_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, H_ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function trek_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.trek_import_64(control::Ptr{trek_control_type{Float64,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                            H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

function trek_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.trek_import_q(control::Ptr{trek_control_type{Float128,Int32}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                           H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                           H_ptr::Ptr{Int32})::Cvoid
end

function trek_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, H_type, H_ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.trek_import_q_64(control::Ptr{trek_control_type{Float128,
                                                                               Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                                 H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                                 H_ptr::Ptr{Int64})::Cvoid
end

export trek_s_import

function trek_s_import(::Type{Float32}, ::Type{Int32}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_single.trek_s_import_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, S_type::Ptr{Cchar}, S_ne::Int32,
                                          S_row::Ptr{Int32}, S_col::Ptr{Int32},
                                          S_ptr::Ptr{Int32})::Cvoid
end

function trek_s_import(::Type{Float32}, ::Type{Int64}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_single_64.trek_s_import_s_64(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, S_type::Ptr{Cchar}, S_ne::Int64,
                                                S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                                S_ptr::Ptr{Int64})::Cvoid
end

function trek_s_import(::Type{Float64}, ::Type{Int32}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_double.trek_s_import(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        S_type::Ptr{Cchar}, S_ne::Int32, S_row::Ptr{Int32},
                                        S_col::Ptr{Int32}, S_ptr::Ptr{Int32})::Cvoid
end

function trek_s_import(::Type{Float64}, ::Type{Int64}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_double_64.trek_s_import_64(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, S_type::Ptr{Cchar}, S_ne::Int64,
                                              S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                              S_ptr::Ptr{Int64})::Cvoid
end

function trek_s_import(::Type{Float128}, ::Type{Int32}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_quadruple.trek_s_import_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, S_type::Ptr{Cchar}, S_ne::Int32,
                                             S_row::Ptr{Int32}, S_col::Ptr{Int32},
                                             S_ptr::Ptr{Int32})::Cvoid
end

function trek_s_import(::Type{Float128}, ::Type{Int64}, data, status, n, S_type, S_ne, S_row,
                      S_col, S_ptr)
  @ccall libgalahad_quadruple_64.trek_s_import_q_64(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64,
                                                   S_type::Ptr{Cchar}, S_ne::Int64,
                                                   S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                                   S_ptr::Ptr{Int64})::Cvoid
end
export trek_reset_control

function trek_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.trek_reset_control_s(control::Ptr{trek_control_type{Float32,Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function trek_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.trek_reset_control_s_64(control::Ptr{trek_control_type{Float32,
                                                                                   Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

function trek_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.trek_reset_control(control::Ptr{trek_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function trek_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.trek_reset_control_64(control::Ptr{trek_control_type{Float64,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

function trek_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.trek_reset_control_q(control::Ptr{trek_control_type{Float128,
                                                                                Int32}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int32})::Cvoid
end

function trek_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.trek_reset_control_q_64(control::Ptr{trek_control_type{Float128,
                                                                                      Int64}},
                                                        data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64})::Cvoid
end

export trek_solve_problem

function trek_solve_problem(::Type{Float32}, ::Type{Int32}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_single.trek_solve_problem_s(data::Ptr{Ptr{Cvoid}}, 
                                                status::Ptr{Int32},
                                                n::Int32, 
                                                H_ne::Int32,
                                                H_val::Ptr{Float32}, 
                                                c::Ptr{Float32}, 
                                                radius::Float32,
                                                x::Ptr{Float32},
                                                S_ne::Int32, 
                                                S_val::Ptr{Float32})::Cvoid
end

function trek_solve_problem(::Type{Float32}, ::Type{Int64}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_single_64.trek_solve_problem_s_64(data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64},
                                                n::Int64, 
                                                H_ne::Int64,
                                                H_val::Ptr{Float32}, 
                                                c::Ptr{Float32}, 
                                                radius::Float32,
                                                x::Ptr{Float32},
                                                S_ne::Int64, 
                                                S_val::Ptr{Float32})::Cvoid
end

function trek_solve_problem(::Type{Float64}, ::Type{Int32}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_double.trek_solve_problem(data::Ptr{Ptr{Cvoid}}, 
                                                status::Ptr{Int32},
                                                n::Int32, 
                                                H_ne::Int32,
                                                H_val::Ptr{Float64}, 
                                                c::Ptr{Float64}, 
                                                radius::Float64,
                                                x::Ptr{Float64},
                                                S_ne::Int32, 
                                                S_val::Ptr{Float64})::Cvoid
end

function trek_solve_problem(::Type{Float64}, ::Type{Int64}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_double_64.trek_solve_problem_64(data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64},
                                                n::Int64, 
                                                H_ne::Int64,
                                                H_val::Ptr{Float64}, 
                                                c::Ptr{Float64}, 
                                                radius::Float64,
                                                x::Ptr{Float64},
                                                S_ne::Int64, 
                                                S_val::Ptr{Float64})::Cvoid
end

function trek_solve_problem(::Type{Float128}, ::Type{Int32}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_quadruple.trek_solve_problem_q(data::Ptr{Ptr{Cvoid}}, 
                                                 status::Ptr{Int32},
                                                 n::Int32, 
                                                 H_ne::Int32,
                                                 H_val::Ptr{Float128}, 
                                                 c::Ptr{Float128}, 
                                                 radius::Cfloat128,
                                                 x::Ptr{Float128},
                                                 S_ne::Int32, 
                                                 S_val::Ptr{Float128})::Cvoid
end

function trek_solve_problem(::Type{Float128}, ::Type{Int64}, data, status,
                            n, H_ne, H_val, c, radius, x, S_ne, S_val)
  @ccall libgalahad_quadruple_64.trek_solve_problem_q_64(data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64},
                                                 n::Int64, 
                                                 H_ne::Int64,
                                                 H_val::Ptr{Float128}, 
                                                 c::Ptr{Float128}, 
                                                 radius::Cfloat128,
                                                 x::Ptr{Float128},
                                                 S_ne::Int64, 
                                                 S_val::Ptr{Float128})::Cvoid
end

export trek_information

function trek_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.trek_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{trek_inform_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function trek_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.trek_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{trek_inform_type{Float32,
                                                                               Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function trek_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.trek_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{trek_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function trek_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.trek_information_64(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{trek_inform_type{Float64,Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

function trek_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.trek_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{trek_inform_type{Float128,Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function trek_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.trek_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{trek_inform_type{Float128,
                                                                                  Int64}},
                                                      status::Ptr{Int64})::Cvoid
end

export trek_terminate

function trek_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.trek_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{trek_control_type{Float32,Int32}},
                                           inform::Ptr{trek_inform_type{Float32,Int32}})::Cvoid
end

function trek_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.trek_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{trek_control_type{Float32,
                                                                               Int64}},
                                                 inform::Ptr{trek_inform_type{Float32,Int64}})::Cvoid
end

function trek_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.trek_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{trek_control_type{Float64,Int32}},
                                         inform::Ptr{trek_inform_type{Float64,Int32}})::Cvoid
end

function trek_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.trek_terminate_64(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{trek_control_type{Float64,Int64}},
                                               inform::Ptr{trek_inform_type{Float64,Int64}})::Cvoid
end

function trek_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.trek_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{trek_control_type{Float128,Int32}},
                                              inform::Ptr{trek_inform_type{Float128,Int32}})::Cvoid
end

function trek_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.trek_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{trek_control_type{Float128,
                                                                                  Int64}},
                                                    inform::Ptr{trek_inform_type{Float128,
                                                                                Int64}})::Cvoid
end

function run_sif(::Val{:trek}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runtrek_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:trek}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runtrek_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
