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

export trs_initialize_s

function trs_initialize_s(data, control, status)
  @ccall libgalahad_single.trs_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{trs_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export trs_initialize

function trs_initialize(data, control, status)
  @ccall libgalahad_double.trs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{trs_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export trs_read_specfile_s

function trs_read_specfile_s(control, specfile)
  @ccall libgalahad_single.trs_read_specfile_s(control::Ptr{trs_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export trs_read_specfile

function trs_read_specfile(control, specfile)
  @ccall libgalahad_double.trs_read_specfile(control::Ptr{trs_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export trs_import_s

function trs_import_s(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.trs_import_s(control::Ptr{trs_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export trs_import

function trs_import(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.trs_import(control::Ptr{trs_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export trs_import_m_s

function trs_import_m_s(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_single.trs_import_m_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                          M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

export trs_import_m

function trs_import_m(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_double.trs_import_m(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                        M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

export trs_import_a_s

function trs_import_a_s(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.trs_import_a_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                          A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                          A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

export trs_import_a

function trs_import_a(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.trs_import_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                        A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                        A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

export trs_reset_control_s

function trs_reset_control_s(control, data, status)
  @ccall libgalahad_single.trs_reset_control_s(control::Ptr{trs_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export trs_reset_control

function trs_reset_control(control, data, status)
  @ccall libgalahad_double.trs_reset_control(control::Ptr{trs_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export trs_solve_problem_s

function trs_solve_problem_s(data, status, n, radius, f, c, H_ne, H_val, x, M_ne, M_val, m,
                             A_ne, A_val, y)
  @ccall libgalahad_single.trs_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, radius::Float32, f::Float32,
                                               c::Ptr{Float32}, H_ne::Cint,
                                               H_val::Ptr{Float32}, x::Ptr{Float32},
                                               M_ne::Cint, M_val::Ptr{Float32}, m::Cint,
                                               A_ne::Cint, A_val::Ptr{Float32},
                                               y::Ptr{Float32})::Cvoid
end

export trs_solve_problem

function trs_solve_problem(data, status, n, radius, f, c, H_ne, H_val, x, M_ne, M_val, m,
                           A_ne, A_val, y)
  @ccall libgalahad_double.trs_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, radius::Float64, f::Float64,
                                             c::Ptr{Float64}, H_ne::Cint,
                                             H_val::Ptr{Float64}, x::Ptr{Float64},
                                             M_ne::Cint, M_val::Ptr{Float64}, m::Cint,
                                             A_ne::Cint, A_val::Ptr{Float64},
                                             y::Ptr{Float64})::Cvoid
end

export trs_information_s

function trs_information_s(data, inform, status)
  @ccall libgalahad_single.trs_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{trs_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export trs_information

function trs_information(data, inform, status)
  @ccall libgalahad_double.trs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{trs_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export trs_terminate_s

function trs_terminate_s(data, control, inform)
  @ccall libgalahad_single.trs_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{trs_control_type{Float32}},
                                           inform::Ptr{trs_inform_type{Float32}})::Cvoid
end

export trs_terminate

function trs_terminate(data, control, inform)
  @ccall libgalahad_double.trs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{trs_control_type{Float64}},
                                         inform::Ptr{trs_inform_type{Float64}})::Cvoid
end
