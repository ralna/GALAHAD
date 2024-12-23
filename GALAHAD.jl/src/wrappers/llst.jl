export llst_control_type

struct llst_control_type{T}
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
end

export llst_time_type

struct llst_time_type{T}
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

export llst_history_type

struct llst_history_type{T}
  lambda::T
  x_norm::T
  r_norm::T
end

export llst_inform_type

struct llst_inform_type{T}
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
end

export llst_initialize

function llst_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.llst_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{llst_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function llst_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.llst_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{llst_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function llst_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.llst_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{llst_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export llst_read_specfile

function llst_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.llst_read_specfile_s(control::Ptr{llst_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function llst_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.llst_read_specfile(control::Ptr{llst_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function llst_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.llst_read_specfile_q(control::Ptr{llst_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export llst_import

function llst_import(::Type{Float32}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_single.llst_import_s(control::Ptr{llst_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                         n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

function llst_import(::Type{Float64}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_double.llst_import(control::Ptr{llst_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                       A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                       A_ptr::Ptr{Cint})::Cvoid
end

function llst_import(::Type{Float128}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_quadruple.llst_import_q(control::Ptr{llst_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            m::Cint, n::Cint, A_type::Ptr{Cchar},
                                            A_ne::Cint, A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                            A_ptr::Ptr{Cint})::Cvoid
end

export llst_import_scaling

function llst_import_scaling(::Type{Float32}, control, data, status, n, S_type, S_ne, S_row,
                             S_col, S_ptr)
  @ccall libgalahad_single.llst_import_scaling_s(control::Ptr{llst_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                                 S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                 S_ptr::Ptr{Cint})::Cvoid
end

function llst_import_scaling(::Type{Float64}, control, data, status, n, S_type, S_ne, S_row,
                             S_col, S_ptr)
  @ccall libgalahad_double.llst_import_scaling(control::Ptr{llst_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                               S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                               S_ptr::Ptr{Cint})::Cvoid
end

function llst_import_scaling(::Type{Float128}, control, data, status, n, S_type, S_ne,
                             S_row, S_col, S_ptr)
  @ccall libgalahad_quadruple.llst_import_scaling_q(control::Ptr{llst_control_type{Float128}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    S_type::Ptr{Cchar}, S_ne::Cint,
                                                    S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                    S_ptr::Ptr{Cint})::Cvoid
end

export llst_reset_control

function llst_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.llst_reset_control_s(control::Ptr{llst_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function llst_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.llst_reset_control(control::Ptr{llst_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function llst_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.llst_reset_control_q(control::Ptr{llst_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export llst_solve_problem

function llst_solve_problem(::Type{Float32}, data, status, m, n, radius, A_ne, A_val, b, x,
                            S_ne, S_val)
  @ccall libgalahad_single.llst_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, radius::Float32,
                                                A_ne::Cint, A_val::Ptr{Float32},
                                                b::Ptr{Float32}, x::Ptr{Float32},
                                                S_ne::Cint, S_val::Ptr{Float32})::Cvoid
end

function llst_solve_problem(::Type{Float64}, data, status, m, n, radius, A_ne, A_val, b, x,
                            S_ne, S_val)
  @ccall libgalahad_double.llst_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, radius::Float64, A_ne::Cint,
                                              A_val::Ptr{Float64}, b::Ptr{Float64},
                                              x::Ptr{Float64}, S_ne::Cint,
                                              S_val::Ptr{Float64})::Cvoid
end

function llst_solve_problem(::Type{Float128}, data, status, m, n, radius, A_ne, A_val, b, x,
                            S_ne, S_val)
  @ccall libgalahad_quadruple.llst_solve_problem_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   m::Cint, n::Cint, radius::Cfloat128,
                                                   A_ne::Cint, A_val::Ptr{Float128},
                                                   b::Ptr{Float128}, x::Ptr{Float128},
                                                   S_ne::Cint, S_val::Ptr{Float128})::Cvoid
end

export llst_information

function llst_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.llst_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{llst_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function llst_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.llst_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{llst_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function llst_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.llst_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{llst_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export llst_terminate

function llst_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.llst_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{llst_control_type{Float32}},
                                            inform::Ptr{llst_inform_type{Float32}})::Cvoid
end

function llst_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.llst_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{llst_control_type{Float64}},
                                          inform::Ptr{llst_inform_type{Float64}})::Cvoid
end

function llst_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.llst_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{llst_control_type{Float128}},
                                               inform::Ptr{llst_inform_type{Float128}})::Cvoid
end
