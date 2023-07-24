export llsr_control_type

mutable struct llsr_control_type{T}
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
  use_initial_multiplier::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T}
  sls_control::sls_control_type{T}
  ir_control::ir_control_type{T}

  function llsr_control_type{T}() where T
    type = new()
    type.sbls_control = sbls_control_type{T}()
    type.sls_control = sls_control_type{T}()
    type.ir_control = ir_control_type{T}()
    return type
  end
end

export llsr_time_type

mutable struct llsr_time_type{T}
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

  llsr_time_type{T}() where T = new()
end

export llsr_history_type

mutable struct llsr_history_type{T}
  lambda::T
  x_norm::T
  r_norm::T

  llsr_history_type{T}() where T = new()
end

export llsr_inform_type

mutable struct llsr_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorizations::Cint
  len_history::Cint
  r_norm::T
  x_norm::T
  multiplier::T
  bad_alloc::NTuple{81,Cchar}
  time::llsr_time_type{T}
  history::NTuple{100,llsr_history_type{T}}
  sbls_inform::sbls_inform_type{T}
  sls_inform::sls_inform_type{T}
  ir_inform::ir_inform_type{T}

  function llsr_inform_type{T}() where T
    type = new()
    type.time = llsr_time_type{T}()
    type.history = ntuple(x -> llsr_history_type{T}(), 100)
    type.sbls_inform = sbls_inform_type{T}()
    type.sls_inform = sls_inform_type{T}()
    type.ir_inform = ir_inform_type{T}()
    return type
  end
end

export llsr_initialize_s

function llsr_initialize_s(data, control, status)
  @ccall libgalahad_single.llsr_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{llsr_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export llsr_initialize

function llsr_initialize(data, control, status)
  @ccall libgalahad_double.llsr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{llsr_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export llsr_read_specfile_s

function llsr_read_specfile_s(control, specfile)
  @ccall libgalahad_single.llsr_read_specfile_s(control::Ref{llsr_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export llsr_read_specfile

function llsr_read_specfile(control, specfile)
  @ccall libgalahad_double.llsr_read_specfile(control::Ref{llsr_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export llsr_import_s

function llsr_import_s(control, data, status, m, n, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.llsr_import_s(control::Ref{llsr_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                         n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

export llsr_import

function llsr_import(control, data, status, m, n, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.llsr_import(control::Ref{llsr_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                       A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                       A_ptr::Ptr{Cint})::Cvoid
end

export llsr_import_scaling_s

function llsr_import_scaling_s(control, data, status, n, S_type, S_ne, S_row, S_col, S_ptr)
  @ccall libgalahad_single.llsr_import_scaling_s(control::Ref{llsr_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                                 S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                 S_ptr::Ptr{Cint})::Cvoid
end

export llsr_import_scaling

function llsr_import_scaling(control, data, status, n, S_type, S_ne, S_row, S_col, S_ptr)
  @ccall libgalahad_double.llsr_import_scaling(control::Ref{llsr_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                               S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                               S_ptr::Ptr{Cint})::Cvoid
end

export llsr_reset_control_s

function llsr_reset_control_s(control, data, status)
  @ccall libgalahad_single.llsr_reset_control_s(control::Ref{llsr_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export llsr_reset_control

function llsr_reset_control(control, data, status)
  @ccall libgalahad_double.llsr_reset_control(control::Ref{llsr_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export llsr_solve_problem_s

function llsr_solve_problem_s(data, status, m, n, power, weight, A_ne, A_val, b, x, S_ne,
                            S_val)
  @ccall libgalahad_single.llsr_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float32,
                                                weight::Float32, A_ne::Cint,
                                                A_val::Ptr{Float32}, b::Ptr{Float32},
                                                x::Ptr{Float32}, S_ne::Cint,
                                                S_val::Ptr{Float32})::Cvoid
end

export llsr_solve_problem

function llsr_solve_problem(data, status, m, n, power, weight, A_ne, A_val, b, x, S_ne,
                          S_val)
  @ccall libgalahad_double.llsr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, power::Float64,
                                              weight::Float64, A_ne::Cint,
                                              A_val::Ptr{Float64}, b::Ptr{Float64},
                                              x::Ptr{Float64}, S_ne::Cint,
                                              S_val::Ptr{Float64})::Cvoid
end

export llsr_information_s

function llsr_information_s(data, inform, status)
  @ccall libgalahad_single.llsr_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{llsr_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export llsr_information

function llsr_information(data, inform, status)
  @ccall libgalahad_double.llsr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{llsr_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export llsr_terminate_s

function llsr_terminate_s(data, control, inform)
  @ccall libgalahad_single.llsr_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{llsr_control_type{Float32}},
                                            inform::Ref{llsr_inform_type{Float32}})::Cvoid
end

export llsr_terminate

function llsr_terminate(data, control, inform)
  @ccall libgalahad_double.llsr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{llsr_control_type{Float64}},
                                          inform::Ref{llsr_inform_type{Float64}})::Cvoid
end
