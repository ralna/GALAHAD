export llst_control_type

mutable struct llst_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  new_a::Cint
  new_s::Cint
  max_factorizations::Cint
  taylor_max_degree::Cint
  initial_multiplier::T
  lower::T
  upper::T
  stop_normal::T
  equality_problem::Bool
  use_initial_multiplier::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T}
  sls_control::sls_control_type{T}
  ir_control::ir_control_type{T}

  llst_control_type{T}() where T = new()
end

export llst_time_type

mutable struct llst_time_type{T}
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

  llst_time_type{T}() where T = new()
end

export llst_history_type

mutable struct llst_history_type{T}
  lambda::T
  x_norm::T
  r_norm::T

  llst_history_type{T}() where T = new()
end

export llst_inform_type

mutable struct llst_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorizations::Cint
  len_history::Cint
  r_norm::T
  x_norm::T
  multiplier::T
  bad_alloc::NTuple{81,Cchar}
  time::llst_time_type{T}
  history::NTuple{100,llst_history_type{T}}
  sbls_inform::sbls_inform_type{T}
  sls_inform::sls_inform_type{T}
  ir_inform::ir_inform_type{T}

  llst_inform_type{T}() where T = new()
end

export llst_initialize_s

function llst_initialize_s(data, control, status)
  @ccall libgalahad_single.llst_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{llst_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export llst_initialize

function llst_initialize(data, control, status)
  @ccall libgalahad_double.llst_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{llst_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export llst_read_specfile_s

function llst_read_specfile_s(control, specfile)
  @ccall libgalahad_single.llst_read_specfile_s(control::Ref{llst_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export llst_read_specfile

function llst_read_specfile(control, specfile)
  @ccall libgalahad_double.llst_read_specfile(control::Ref{llst_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export llst_import_s

function llst_import_s(control, data, status, m, n, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.llst_import_s(control::Ref{llst_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                         n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

export llst_import

function llst_import(control, data, status, m, n, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.llst_import(control::Ref{llst_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                       A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                       A_ptr::Ptr{Cint})::Cvoid
end

export llst_import_scaling_s

function llst_import_scaling_s(control, data, status, n, S_type, S_ne, S_row, S_col, S_ptr)
  @ccall libgalahad_single.llst_import_scaling_s(control::Ref{llst_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                                 S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                 S_ptr::Ptr{Cint})::Cvoid
end

export llst_import_scaling

function llst_import_scaling(control, data, status, n, S_type, S_ne, S_row, S_col, S_ptr)
  @ccall libgalahad_double.llst_import_scaling(control::Ref{llst_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                               S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                               S_ptr::Ptr{Cint})::Cvoid
end

export llst_reset_control_s

function llst_reset_control_s(control, data, status)
  @ccall libgalahad_single.llst_reset_control_s(control::Ref{llst_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export llst_reset_control

function llst_reset_control(control, data, status)
  @ccall libgalahad_double.llst_reset_control(control::Ref{llst_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export llst_solve_problem_s

function llst_solve_problem_s(data, status, m, n, radius, A_ne, A_val, b, x, S_ne, S_val)
  @ccall libgalahad_single.llst_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, radius::Float32,
                                                A_ne::Cint, A_val::Ptr{Float32},
                                                b::Ptr{Float32}, x::Ptr{Float32},
                                                S_ne::Cint, S_val::Ptr{Float32})::Cvoid
end

export llst_solve_problem

function llst_solve_problem(data, status, m, n, radius, A_ne, A_val, b, x, S_ne, S_val)
  @ccall libgalahad_double.llst_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, radius::Float64,
                                              A_ne::Cint, A_val::Ptr{Float64},
                                              b::Ptr{Float64}, x::Ptr{Float64},
                                              S_ne::Cint, S_val::Ptr{Float64})::Cvoid
end

export llst_information_s

function llst_information_s(data, inform, status)
  @ccall libgalahad_single.llst_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{llst_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export llst_information

function llst_information(data, inform, status)
  @ccall libgalahad_double.llst_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{llst_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export llst_terminate_s

function llst_terminate_s(data, control, inform)
  @ccall libgalahad_single.llst_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{llst_control_type{Float32}},
                                            inform::Ref{llst_inform_type{Float32}})::Cvoid
end

export llst_terminate

function llst_terminate(data, control, inform)
  @ccall libgalahad_double.llst_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{llst_control_type{Float64}},
                                          inform::Ref{llst_inform_type{Float64}})::Cvoid
end
