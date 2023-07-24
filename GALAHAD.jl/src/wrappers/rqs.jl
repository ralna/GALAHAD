export rqs_control_type

mutable struct rqs_control_type{T}
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
  stop_hard::T
  start_invit_tol::T
  start_invitmax_tol::T
  use_initial_multiplier::Bool
  initialize_approx_eigenvector::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  problem_file::NTuple{31,Cchar}
  symmetric_linear_solver::NTuple{31,Cchar}
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
  ir_control::ir_control_type{T}

  function rqs_control_type{T}() where T
    type = new()
    type.sls_control = sls_control_type{T}()
    type.ir_control = ir_control_type{T}()
    return type
  end
end

export rqs_time_type

mutable struct rqs_time_type{T}
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

  rqs_time_type{T}() where T = new()
end

export rqs_history_type

mutable struct rqs_history_type{T}
  lambda::T
  x_norm::T

  rqs_history_type{T}() where T = new()
end

export rqs_inform_type

mutable struct rqs_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorizations::Cint
  max_entries_factors::Int64
  len_history::Cint
  obj::T
  obj_regularized::T
  x_norm::T
  multiplier::T
  pole::T
  dense_factorization::Bool
  hard_case::Bool
  bad_alloc::NTuple{81,Cchar}
  time::rqs_time_type{T}
  history::NTuple{100,rqs_history_type{T}}
  sls_inform::sls_inform_type{T}
  ir_inform::ir_inform_type{T}

  function rqs_inform_type{T}() where T
    type = new()
    type.time = rqs_time_type{T}()
    type.history = ntuple(x -> rqs_history_type{T}(), 100)
    type.sls_inform = sls_inform_type{T}()
    type.ir_inform = ir_inform_type{T}()
    return type
  end
end

export rqs_initialize_s

function rqs_initialize_s(data, control, status)
  @ccall libgalahad_single.rqs_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{rqs_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export rqs_initialize

function rqs_initialize(data, control, status)
  @ccall libgalahad_double.rqs_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{rqs_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export rqs_read_specfile_s

function rqs_read_specfile_s(control, specfile)
  @ccall libgalahad_single.rqs_read_specfile_s(control::Ref{rqs_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export rqs_read_specfile

function rqs_read_specfile(control, specfile)
  @ccall libgalahad_double.rqs_read_specfile(control::Ref{rqs_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export rqs_import_s

function rqs_import_s(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.rqs_import_s(control::Ref{rqs_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export rqs_import

function rqs_import(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.rqs_import(control::Ref{rqs_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export rqs_import_m_s

function rqs_import_m_s(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_single.rqs_import_m_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                          M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

export rqs_import_m

function rqs_import_m(data, status, n, M_type, M_ne, M_row, M_col, M_ptr)
  @ccall libgalahad_double.rqs_import_m(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        M_type::Ptr{Cchar}, M_ne::Cint, M_row::Ptr{Cint},
                                        M_col::Ptr{Cint}, M_ptr::Ptr{Cint})::Cvoid
end

export rqs_import_a_s

function rqs_import_a_s(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.rqs_import_a_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                          A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                          A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

export rqs_import_a

function rqs_import_a(data, status, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.rqs_import_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                        A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                        A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

export rqs_reset_control_s

function rqs_reset_control_s(control, data, status)
  @ccall libgalahad_single.rqs_reset_control_s(control::Ref{rqs_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export rqs_reset_control

function rqs_reset_control(control, data, status)
  @ccall libgalahad_double.rqs_reset_control(control::Ref{rqs_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export rqs_solve_problem_s

function rqs_solve_problem_s(data, status, n, power, weight, f, c, H_ne, H_val, x, M_ne,
                           M_val, m, A_ne, A_val, y)
  @ccall libgalahad_single.rqs_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, power::Float32, weight::Float32,
                                               f::Float32, c::Ptr{Float32}, H_ne::Cint,
                                               H_val::Ptr{Float32}, x::Ptr{Float32},
                                               M_ne::Cint, M_val::Ptr{Float32}, m::Cint,
                                               A_ne::Cint, A_val::Ptr{Float32},
                                               y::Ptr{Float32})::Cvoid
end

export rqs_solve_problem

function rqs_solve_problem(data, status, n, power, weight, f, c, H_ne, H_val, x, M_ne,
                         M_val, m, A_ne, A_val, y)
  @ccall libgalahad_double.rqs_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, power::Float64, weight::Float64,
                                             f::Float64, c::Ptr{Float64}, H_ne::Cint,
                                             H_val::Ptr{Float64}, x::Ptr{Float64},
                                             M_ne::Cint, M_val::Ptr{Float64}, m::Cint,
                                             A_ne::Cint, A_val::Ptr{Float64},
                                             y::Ptr{Float64})::Cvoid
end

export rqs_information_s

function rqs_information_s(data, inform, status)
  @ccall libgalahad_single.rqs_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{rqs_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export rqs_information

function rqs_information(data, inform, status)
  @ccall libgalahad_double.rqs_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{rqs_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export rqs_terminate_s

function rqs_terminate_s(data, control, inform)
  @ccall libgalahad_single.rqs_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{rqs_control_type{Float32}},
                                           inform::Ref{rqs_inform_type{Float32}})::Cvoid
end

export rqs_terminate

function rqs_terminate(data, control, inform)
  @ccall libgalahad_double.rqs_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{rqs_control_type{Float64}},
                                         inform::Ref{rqs_inform_type{Float64}})::Cvoid
end
