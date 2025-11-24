export nrek_control_type

struct nrek_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  eks_max::INT
  it_max::INT
  f::T
  increase::T
  stop_residual::T
  reorthogonalize::Bool
  s_version_52::Bool
  perturb_c::Bool
  stop_check_all_orders::Bool
  new_weight::Bool
  new_values::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  linear_solver::NTuple{31,Cchar}
  linear_solver_for_s::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
  sls_s_control::sls_control_type{T,INT}
  rqs_control::rqs_control_type{T,INT}
end

export nrek_time_type

struct nrek_time_type{T}
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

export nrek_inform_type

struct nrek_inform_type{T,INT}
  status::INT
  alloc_status::INT
  iter::INT
  n_vec::INT
  obj::T
  obj_regularized::T
  x_norm::T
  multiplier::T
  weight::T
  next_weight::T
  error::T
  bad_alloc::NTuple{81,Cchar}
  time::nrek_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  sls_s_inform::sls_inform_type{T,INT}
  rqs_inform::rqs_inform_type{T,INT}
end

export nrek_initialize

function nrek_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.nrek_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{nrek_control_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nrek_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.nrek_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{nrek_control_type{Float32,
                                                                                  Int64}},
                                                   status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nrek_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.nrek_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{nrek_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nrek_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.nrek_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{nrek_control_type{Float64,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nrek_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.nrek_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nrek_control_type{Float128,
                                                                               Int32}},
                                                status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function nrek_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.nrek_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{nrek_control_type{Float128,
                                                                                     Int64}},
                                                      status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export nrek_read_specfile

function nrek_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.nrek_read_specfile_s(control::Ptr{nrek_control_type{Float32,
                                                                               Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function nrek_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.nrek_read_specfile_s_64(control::Ptr{nrek_control_type{Float32,
                                                                                     Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

function nrek_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.nrek_read_specfile(control::Ptr{nrek_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function nrek_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.nrek_read_specfile_64(control::Ptr{nrek_control_type{Float64,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function nrek_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.nrek_read_specfile_q(control::Ptr{nrek_control_type{Float128,
                                                                                  Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function nrek_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.nrek_read_specfile_q_64(control::Ptr{nrek_control_type{Float128,
                                                                                        Int64}},
                                                         specfile::Ptr{Cchar})::Cvoid
end

export nrek_import

function nrek_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, H_type, H_ne,
                     H_row, H_col, H_ptr)
  @ccall libgalahad_single.nrek_import_s(control::Ptr{nrek_control_type{Float32,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                         H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32})::Cvoid
end

function nrek_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, H_type, H_ne,
                     H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.nrek_import_s_64(control::Ptr{nrek_control_type{Float32,
                                                                              Int64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                               H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                               H_ptr::Ptr{Int64})::Cvoid
end

function nrek_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, H_type, H_ne,
                     H_row, H_col, H_ptr)
  @ccall libgalahad_double.nrek_import(control::Ptr{nrek_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       H_type::Ptr{Cchar}, H_ne::Int32, H_row::Ptr{Int32},
                                       H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function nrek_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, H_type, H_ne,
                     H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.nrek_import_64(control::Ptr{nrek_control_type{Float64,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                             H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                             H_ptr::Ptr{Int64})::Cvoid
end

function nrek_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, H_type,
                     H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.nrek_import_q(control::Ptr{nrek_control_type{Float128,Int32}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            n::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                            H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                            H_ptr::Ptr{Int32})::Cvoid
end

function nrek_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, H_type,
                     H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.nrek_import_q_64(control::Ptr{nrek_control_type{Float128,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                  n::Int64, H_type::Ptr{Cchar}, H_ne::Int64,
                                                  H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                                  H_ptr::Ptr{Int64})::Cvoid
end

export nrek_s_import

function nrek_s_import(::Type{Float32}, ::Type{Int32}, data, status, n, S_type, S_ne, S_row,
                       S_col, S_ptr)
  @ccall libgalahad_single.nrek_s_import_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, S_type::Ptr{Cchar}, S_ne::Int32,
                                           S_row::Ptr{Int32}, S_col::Ptr{Int32},
                                           S_ptr::Ptr{Int32})::Cvoid
end

function nrek_s_import(::Type{Float32}, ::Type{Int64}, data, status, n, S_type, S_ne, S_row,
                       S_col, S_ptr)
  @ccall libgalahad_single_64.nrek_s_import_s_64(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, S_type::Ptr{Cchar}, S_ne::Int64,
                                                 S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                                 S_ptr::Ptr{Int64})::Cvoid
end

function nrek_s_import(::Type{Float64}, ::Type{Int32}, data, status, n, S_type, S_ne, S_row,
                       S_col, S_ptr)
  @ccall libgalahad_double.nrek_s_import(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, S_type::Ptr{Cchar}, S_ne::Int32,
                                         S_row::Ptr{Int32}, S_col::Ptr{Int32},
                                         S_ptr::Ptr{Int32})::Cvoid
end

function nrek_s_import(::Type{Float64}, ::Type{Int64}, data, status, n, S_type, S_ne, S_row,
                       S_col, S_ptr)
  @ccall libgalahad_double_64.nrek_s_import_64(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               n::Int64, S_type::Ptr{Cchar}, S_ne::Int64,
                                               S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                               S_ptr::Ptr{Int64})::Cvoid
end

function nrek_s_import(::Type{Float128}, ::Type{Int32}, data, status, n, S_type, S_ne,
                       S_row, S_col, S_ptr)
  @ccall libgalahad_quadruple.nrek_s_import_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, S_type::Ptr{Cchar}, S_ne::Int32,
                                              S_row::Ptr{Int32}, S_col::Ptr{Int32},
                                              S_ptr::Ptr{Int32})::Cvoid
end

function nrek_s_import(::Type{Float128}, ::Type{Int64}, data, status, n, S_type, S_ne,
                       S_row, S_col, S_ptr)
  @ccall libgalahad_quadruple_64.nrek_s_import_q_64(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    S_type::Ptr{Cchar}, S_ne::Int64,
                                                    S_row::Ptr{Int64}, S_col::Ptr{Int64},
                                                    S_ptr::Ptr{Int64})::Cvoid
end

export nrek_reset_control

function nrek_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.nrek_reset_control_s(control::Ptr{nrek_control_type{Float32,
                                                                               Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function nrek_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.nrek_reset_control_s_64(control::Ptr{nrek_control_type{Float32,
                                                                                     Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64})::Cvoid
end

function nrek_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.nrek_reset_control(control::Ptr{nrek_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function nrek_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.nrek_reset_control_64(control::Ptr{nrek_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

function nrek_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.nrek_reset_control_q(control::Ptr{nrek_control_type{Float128,
                                                                                  Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32})::Cvoid
end

function nrek_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.nrek_reset_control_q_64(control::Ptr{nrek_control_type{Float128,
                                                                                        Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64})::Cvoid
end

export nrek_solve_problem

function nrek_solve_problem(::Type{Float32}, ::Type{Int32}, data, status, n, H_ne, H_val, c,
                            power, weight, x, S_ne, S_val)
  @ccall libgalahad_single.nrek_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, H_ne::Int32, H_val::Ptr{Float32},
                                                c::Ptr{Float32}, power::Float32,
                                                weight::Float32, x::Ptr{Float32},
                                                S_ne::Int32, S_val::Ptr{Float32})::Cvoid
end

function nrek_solve_problem(::Type{Float32}, ::Type{Int64}, data, status, n, H_ne, H_val, c,
                            power, weight, x, S_ne, S_val)
  @ccall libgalahad_single_64.nrek_solve_problem_s_64(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64,
                                                      H_ne::Int64, H_val::Ptr{Float32},
                                                      c::Ptr{Float32}, power::Float32,
                                                      weight::Float32, x::Ptr{Float32},
                                                      S_ne::Int64,
                                                      S_val::Ptr{Float32})::Cvoid
end

function nrek_solve_problem(::Type{Float64}, ::Type{Int32}, data, status, n, H_ne, H_val, c,
                            power, weight, x, S_ne, S_val)
  @ccall libgalahad_double.nrek_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, H_ne::Int32, H_val::Ptr{Float64},
                                              c::Ptr{Float64}, power::Float64,
                                              weight::Float64, x::Ptr{Float64}, S_ne::Int32,
                                              S_val::Ptr{Float64})::Cvoid
end

function nrek_solve_problem(::Type{Float64}, ::Type{Int64}, data, status, n, H_ne, H_val, c,
                            power, weight, x, S_ne, S_val)
  @ccall libgalahad_double_64.nrek_solve_problem_64(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    H_ne::Int64, H_val::Ptr{Float64},
                                                    c::Ptr{Float64}, power::Float64,
                                                    weight::Float64, x::Ptr{Float64},
                                                    S_ne::Int64, S_val::Ptr{Float64})::Cvoid
end

function nrek_solve_problem(::Type{Float128}, ::Type{Int32}, data, status, n, H_ne, H_val,
                            c, power, weight, x, S_ne, S_val)
  @ccall libgalahad_quadruple.nrek_solve_problem_q(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32,
                                                   H_ne::Int32, H_val::Ptr{Float128},
                                                   c::Ptr{Float128}, power::Cfloat128,
                                                   weight::Cfloat128, x::Ptr{Float128},
                                                   S_ne::Int32, S_val::Ptr{Float128})::Cvoid
end

function nrek_solve_problem(::Type{Float128}, ::Type{Int64}, data, status, n, H_ne, H_val,
                            c, power, weight, x, S_ne, S_val)
  @ccall libgalahad_quadruple_64.nrek_solve_problem_q_64(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, n::Int64,
                                                         H_ne::Int64, H_val::Ptr{Float128},
                                                         c::Ptr{Float128}, power::Cfloat128,
                                                         weight::Cfloat128,
                                                         x::Ptr{Float128}, S_ne::Int64,
                                                         S_val::Ptr{Float128})::Cvoid
end

export nrek_information

function nrek_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.nrek_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{nrek_inform_type{Float32,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function nrek_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.nrek_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{nrek_inform_type{Float32,
                                                                                 Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function nrek_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.nrek_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{nrek_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function nrek_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.nrek_information_64(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{nrek_inform_type{Float64,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function nrek_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.nrek_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{nrek_inform_type{Float128,
                                                                              Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function nrek_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.nrek_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                       inform::Ptr{nrek_inform_type{Float128,
                                                                                    Int64}},
                                                       status::Ptr{Int64})::Cvoid
end

export nrek_terminate

function nrek_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.nrek_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{nrek_control_type{Float32,Int32}},
                                            inform::Ptr{nrek_inform_type{Float32,Int32}})::Cvoid
end

function nrek_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.nrek_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{nrek_control_type{Float32,
                                                                                 Int64}},
                                                  inform::Ptr{nrek_inform_type{Float32,
                                                                               Int64}})::Cvoid
end

function nrek_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.nrek_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{nrek_control_type{Float64,Int32}},
                                          inform::Ptr{nrek_inform_type{Float64,Int32}})::Cvoid
end

function nrek_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.nrek_terminate_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{nrek_control_type{Float64,
                                                                               Int64}},
                                                inform::Ptr{nrek_inform_type{Float64,Int64}})::Cvoid
end

function nrek_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.nrek_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{nrek_control_type{Float128,
                                                                              Int32}},
                                               inform::Ptr{nrek_inform_type{Float128,Int32}})::Cvoid
end

function nrek_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.nrek_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{nrek_control_type{Float128,
                                                                                    Int64}},
                                                     inform::Ptr{nrek_inform_type{Float128,
                                                                                  Int64}})::Cvoid
end

function run_sif(::Val{:nrek}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runnrek_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:nrek}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runnrek_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
