export ssls_control_type

struct ssls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  space_critical::Bool
  deallocate_error_fatal::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
end

export ssls_time_type

struct ssls_time_type{T}
  total::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export ssls_inform_type

struct ssls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  factorization_integer::Int64
  factorization_real::Int64
  rank::INT
  rank_def::Bool
  time::ssls_time_type{T}
  sls_inform::sls_inform_type{T,INT}
end

export ssls_initialize

function ssls_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.ssls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ssls_control_type{Float32,Int32}},
                                             status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function ssls_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.ssls_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{ssls_control_type{Float32,
                                                                                  Int64}},
                                                   status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function ssls_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.ssls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ssls_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function ssls_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.ssls_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{ssls_control_type{Float64,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function ssls_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.ssls_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{ssls_control_type{Float128,
                                                                               Int32}},
                                                status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function ssls_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.ssls_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{ssls_control_type{Float128,
                                                                                     Int64}},
                                                      status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export ssls_read_specfile

function ssls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.ssls_read_specfile_s(control::Ptr{ssls_control_type{Float32,
                                                                               Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function ssls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.ssls_read_specfile_s_64(control::Ptr{ssls_control_type{Float32,
                                                                                     Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

function ssls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.ssls_read_specfile(control::Ptr{ssls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function ssls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.ssls_read_specfile_64(control::Ptr{ssls_control_type{Float64,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function ssls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.ssls_read_specfile_q(control::Ptr{ssls_control_type{Float128,
                                                                                  Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function ssls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.ssls_read_specfile_q_64(control::Ptr{ssls_control_type{Float128,
                                                                                        Int64}},
                                                         specfile::Ptr{Cchar})::Cvoid
end

export ssls_import

function ssls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_single.ssls_import_s(control::Ptr{ssls_control_type{Float32,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, m::Int32, H_type::Ptr{Cchar},
                                         H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                         A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                         A_ptr::Ptr{Int32}, C_type::Ptr{Cchar}, C_ne::Int32,
                                         C_row::Ptr{Int32}, C_col::Ptr{Int32},
                                         C_ptr::Ptr{Int32})::Cvoid
end

function ssls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_single_64.ssls_import_s_64(control::Ptr{ssls_control_type{Float32,
                                                                              Int64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               n::Int64, m::Int64, H_type::Ptr{Cchar},
                                               H_ne::Int64, H_row::Ptr{Int64},
                                               H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                               A_type::Ptr{Cchar}, A_ne::Int64,
                                               A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                               A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                               C_ne::Int64, C_row::Ptr{Int64},
                                               C_col::Ptr{Int64}, C_ptr::Ptr{Int64})::Cvoid
end

function ssls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_double.ssls_import(control::Ptr{ssls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                       H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                       H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                       A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                       A_ptr::Ptr{Int32}, C_type::Ptr{Cchar}, C_ne::Int32,
                                       C_row::Ptr{Int32}, C_col::Ptr{Int32},
                                       C_ptr::Ptr{Int32})::Cvoid
end

function ssls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_double_64.ssls_import_64(control::Ptr{ssls_control_type{Float64,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, H_type::Ptr{Cchar},
                                             H_ne::Int64, H_row::Ptr{Int64},
                                             H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                             A_type::Ptr{Cchar}, A_ne::Int64,
                                             A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                             A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                             C_ne::Int64, C_row::Ptr{Int64},
                                             C_col::Ptr{Int64}, C_ptr::Ptr{Int64})::Cvoid
end

function ssls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_quadruple.ssls_import_q(control::Ptr{ssls_control_type{Float128,Int32}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            n::Int32, m::Int32, H_type::Ptr{Cchar},
                                            H_ne::Int32, H_row::Ptr{Int32},
                                            H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                            A_type::Ptr{Cchar}, A_ne::Int32,
                                            A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                            A_ptr::Ptr{Int32}, C_type::Ptr{Cchar},
                                            C_ne::Int32, C_row::Ptr{Int32},
                                            C_col::Ptr{Int32}, C_ptr::Ptr{Int32})::Cvoid
end

function ssls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, H_type,
                     H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, C_type,
                     C_ne, C_row, C_col, C_ptr)
  @ccall libgalahad_quadruple_64.ssls_import_q_64(control::Ptr{ssls_control_type{Float128,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                  n::Int64, m::Int64, H_type::Ptr{Cchar},
                                                  H_ne::Int64, H_row::Ptr{Int64},
                                                  H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                                  A_type::Ptr{Cchar}, A_ne::Int64,
                                                  A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                                  A_ptr::Ptr{Int64}, C_type::Ptr{Cchar},
                                                  C_ne::Int64, C_row::Ptr{Int64},
                                                  C_col::Ptr{Int64},
                                                  C_ptr::Ptr{Int64})::Cvoid
end

export ssls_reset_control

function ssls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.ssls_reset_control_s(control::Ptr{ssls_control_type{Float32,
                                                                               Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function ssls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.ssls_reset_control_s_64(control::Ptr{ssls_control_type{Float32,
                                                                                     Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64})::Cvoid
end

function ssls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.ssls_reset_control(control::Ptr{ssls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function ssls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.ssls_reset_control_64(control::Ptr{ssls_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

function ssls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.ssls_reset_control_q(control::Ptr{ssls_control_type{Float128,
                                                                                  Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32})::Cvoid
end

function ssls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.ssls_reset_control_q_64(control::Ptr{ssls_control_type{Float128,
                                                                                        Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64})::Cvoid
end

export ssls_factorize_matrix

function ssls_factorize_matrix(::Type{Float32}, ::Type{Int32}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_single.ssls_factorize_matrix_s(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, h_ne::Int32,
                                                   H_val::Ptr{Float32}, a_ne::Int32,
                                                   A_val::Ptr{Float32}, c_ne::Int32,
                                                   C_val::Ptr{Float32})::Cvoid
end

function ssls_factorize_matrix(::Type{Float32}, ::Type{Int64}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_single_64.ssls_factorize_matrix_s_64(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, h_ne::Int64,
                                                         H_val::Ptr{Float32}, a_ne::Int64,
                                                         A_val::Ptr{Float32}, c_ne::Int64,
                                                         C_val::Ptr{Float32})::Cvoid
end

function ssls_factorize_matrix(::Type{Float64}, ::Type{Int32}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_double.ssls_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 h_ne::Int32, H_val::Ptr{Float64},
                                                 a_ne::Int32, A_val::Ptr{Float64},
                                                 c_ne::Int32, C_val::Ptr{Float64})::Cvoid
end

function ssls_factorize_matrix(::Type{Float64}, ::Type{Int64}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_double_64.ssls_factorize_matrix_64(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, h_ne::Int64,
                                                       H_val::Ptr{Float64}, a_ne::Int64,
                                                       A_val::Ptr{Float64}, c_ne::Int64,
                                                       C_val::Ptr{Float64})::Cvoid
end

function ssls_factorize_matrix(::Type{Float128}, ::Type{Int32}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_quadruple.ssls_factorize_matrix_q(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, h_ne::Int32,
                                                      H_val::Ptr{Float128}, a_ne::Int32,
                                                      A_val::Ptr{Float128}, c_ne::Int32,
                                                      C_val::Ptr{Float128})::Cvoid
end

function ssls_factorize_matrix(::Type{Float128}, ::Type{Int64}, data, status, h_ne, H_val,
                               a_ne, A_val, c_ne, C_val)
  @ccall libgalahad_quadruple_64.ssls_factorize_matrix_q_64(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64}, h_ne::Int64,
                                                            H_val::Ptr{Float128},
                                                            a_ne::Int64,
                                                            A_val::Ptr{Float128},
                                                            c_ne::Int64,
                                                            C_val::Ptr{Float128})::Cvoid
end

export ssls_solve_system

function ssls_solve_system(::Type{Float32}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_single.ssls_solve_system_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                               n::Int32, m::Int32, sol::Ptr{Float32})::Cvoid
end

function ssls_solve_system(::Type{Float32}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_single_64.ssls_solve_system_s_64(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64}, n::Int64, m::Int64,
                                                     sol::Ptr{Float32})::Cvoid
end

function ssls_solve_system(::Type{Float64}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_double.ssls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, m::Int32, sol::Ptr{Float64})::Cvoid
end

function ssls_solve_system(::Type{Float64}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_double_64.ssls_solve_system_64(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64, m::Int64,
                                                   sol::Ptr{Float64})::Cvoid
end

function ssls_solve_system(::Type{Float128}, ::Type{Int32}, data, status, n, m, sol)
  @ccall libgalahad_quadruple.ssls_solve_system_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                  n::Int32, m::Int32,
                                                  sol::Ptr{Float128})::Cvoid
end

function ssls_solve_system(::Type{Float128}, ::Type{Int64}, data, status, n, m, sol)
  @ccall libgalahad_quadruple_64.ssls_solve_system_q_64(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n::Int64,
                                                        m::Int64, sol::Ptr{Float128})::Cvoid
end

export ssls_information

function ssls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.ssls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{ssls_inform_type{Float32,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function ssls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.ssls_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{ssls_inform_type{Float32,
                                                                                 Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function ssls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.ssls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{ssls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function ssls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.ssls_information_64(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{ssls_inform_type{Float64,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function ssls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.ssls_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{ssls_inform_type{Float128,
                                                                              Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function ssls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.ssls_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                       inform::Ptr{ssls_inform_type{Float128,
                                                                                    Int64}},
                                                       status::Ptr{Int64})::Cvoid
end

export ssls_terminate

function ssls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.ssls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ssls_control_type{Float32,Int32}},
                                            inform::Ptr{ssls_inform_type{Float32,Int32}})::Cvoid
end

function ssls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.ssls_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{ssls_control_type{Float32,
                                                                                 Int64}},
                                                  inform::Ptr{ssls_inform_type{Float32,
                                                                               Int64}})::Cvoid
end

function ssls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.ssls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ssls_control_type{Float64,Int32}},
                                          inform::Ptr{ssls_inform_type{Float64,Int32}})::Cvoid
end

function ssls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.ssls_terminate_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{ssls_control_type{Float64,
                                                                               Int64}},
                                                inform::Ptr{ssls_inform_type{Float64,Int64}})::Cvoid
end

function ssls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.ssls_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ssls_control_type{Float128,
                                                                              Int32}},
                                               inform::Ptr{ssls_inform_type{Float128,Int32}})::Cvoid
end

function ssls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.ssls_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{ssls_control_type{Float128,
                                                                                    Int64}},
                                                     inform::Ptr{ssls_inform_type{Float128,
                                                                                  Int64}})::Cvoid
end

function run_sif(::Val{:ssls}, ::Val{:single}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runssls_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:ssls}, ::Val{:double}, path_libsif::String, path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runssls_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
