export trs_control_type

struct trs_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  problem::Cint
  print_level::Cint
  dense_factorization::Cint
  new_h::Cint
  new_m::Cint
  new_a::Cint
  max_factorizations::Cint
  inverse_itmax::Cint
  taylor_max_degree::Cint
  initial_multiplier::T
  lower::T
  upper::T
  stop_normal::T
  stop_absolute_normal::T
  stop_hard::T
  start_invit_tol::T
  start_invitmax_tol::T
  equality_problem::Bool
  use_initial_multiplier::Bool
  initialize_approx_eigenvector::Bool
  force_Newton::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  problem_file::NTuple{31,Cchar}
  symmetric_linear_solver::NTuple{31,Cchar}
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
  ir_control::ir_control_type{T}
end

export trs_time_type

struct trs_time_type{T}
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

export trs_history_type

struct trs_history_type{T}
  lambda::T
  x_norm::T
end

export trs_inform_type

struct trs_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorizations::Cint
  max_entries_factors::Int64
  len_history::Cint
  obj::T
  x_norm::T
  multiplier::T
  pole::T
  dense_factorization::Bool
  hard_case::Bool
  bad_alloc::NTuple{81,Cchar}
  time::trs_time_type{T}
  history::NTuple{100,trs_history_type{T}}
  sls_inform::sls_inform_type{T}
  ir_inform::ir_inform_type{T}
end

export trs_initialize

function trs_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.trs_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trs_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function trs_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.trs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{trs_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function trs_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.trs_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{trs_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export trs_read_specfile

function trs_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.trs_read_specfile_s(control::Ptr{trs_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function trs_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.trs_read_specfile(control::Ptr{trs_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function trs_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.trs_read_specfile_q(control::Ptr{trs_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export trs_import

function trs_import(::Type{Float32}, control, data, status, n, H_type, H_ne, H_row, H_col,
                    H_ptr)
  @ccall libgalahad_single.trs_import_s(control::Ptr{trs_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function trs_import(::Type{Float64}, control, data, status, n, H_type, H_ne, H_row, H_col,
                    H_ptr)
  @ccall libgalahad_double.trs_import(control::Ptr{trs_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function trs_import(::Type{Float128}, control, data, status, n, H_type, H_ne, H_row, H_col,
                    H_ptr)
  @ccall libgalahad_quadruple.trs_import_q(control::Ptr{trs_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                           H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                           H_ptr::Ptr{Cint})::Cvoid
end

export trs_import_m

function trs_import_m(::Type{Float32}, data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_single.trs_import_m_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                          M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

function trs_import_m(::Type{Float64}, data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_double.trs_import_m(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                        M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

function trs_import_m(::Type{Float128}, data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_quadruple.trs_import_m_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, M_type::Ptr{Cchar}, M_ne::Cint,
                                             M_row::Ptr{Cint}, M_col::Ptr{Cint},
                                             M_ptr::Ptr{Cint})::Cvoid
end

export trs_import_a

function trs_import_a(::Type{Float32}, data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.trs_import_a_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                          A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                          A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function trs_import_a(::Type{Float64}, data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.trs_import_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                        A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                        A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function trs_import_a(::Type{Float128}, data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.trs_import_a_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                             A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                             A_ptr::Ptr{Cint})::Cvoid
end

export trs_reset_control

function trs_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.trs_reset_control_s(control::Ptr{trs_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function trs_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.trs_reset_control(control::Ptr{trs_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function trs_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.trs_reset_control_q(control::Ptr{trs_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export trs_solve_problem

function trs_solve_problem(::Type{Float32}, data, status, n, radius, f, c, H_ne, H_val, x,
                           M_ne, M_val, m, A_ne, A_val, y)
  @ccall libgalahad_single.trs_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, radius::Float32, f::Float32,
                                               c::Ptr{Float32}, H_ne::Cint,
                                               H_val::Ptr{Float32}, x::Ptr{Float32},
                                               M_ne::Cint, M_val::Ptr{Float32}, m::Cint,
                                               A_ne::Cint, A_val::Ptr{Float32},
                                               y::Ptr{Float32})::Cvoid
end

function trs_solve_problem(::Type{Float64}, data, status, n, radius, f, c, H_ne, H_val, x,
                           M_ne, M_val, m, A_ne, A_val, y)
  @ccall libgalahad_double.trs_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, radius::Float64, f::Float64,
                                             c::Ptr{Float64}, H_ne::Cint,
                                             H_val::Ptr{Float64}, x::Ptr{Float64},
                                             M_ne::Cint, M_val::Ptr{Float64}, m::Cint,
                                             A_ne::Cint, A_val::Ptr{Float64},
                                             y::Ptr{Float64})::Cvoid
end

function trs_solve_problem(::Type{Float128}, data, status, n, radius, f, c, H_ne, H_val, x,
                           M_ne, M_val, m, A_ne, A_val, y)
  @ccall libgalahad_quadruple.trs_solve_problem_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, radius::Float128, f::Float128,
                                                  c::Ptr{Float128}, H_ne::Cint,
                                                  H_val::Ptr{Float128}, x::Ptr{Float128},
                                                  M_ne::Cint, M_val::Ptr{Float128}, m::Cint,
                                                  A_ne::Cint, A_val::Ptr{Float128},
                                                  y::Ptr{Float128})::Cvoid
end

export trs_information

function trs_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.trs_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{trs_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function trs_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.trs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{trs_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function trs_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.trs_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{trs_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export trs_terminate

function trs_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.trs_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{trs_control_type{Float32}},
                                           inform::Ptr{trs_inform_type{Float32}})::Cvoid
end

function trs_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.trs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{trs_control_type{Float64}},
                                         inform::Ptr{trs_inform_type{Float64}})::Cvoid
end

function trs_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.trs_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{trs_control_type{Float128}},
                                              inform::Ptr{trs_inform_type{Float128}})::Cvoid
end
