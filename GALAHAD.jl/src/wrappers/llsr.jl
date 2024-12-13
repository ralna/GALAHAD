export llsr_control_type

struct llsr_control_type{T}
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
end

export llsr_time_type

struct llsr_time_type{T}
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

export llsr_history_type

struct llsr_history_type{T}
  lambda::T
  x_norm::T
  r_norm::T
end

export llsr_inform_type

struct llsr_inform_type{T}
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
end

export llsr_initialize

function llsr_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.llsr_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{llsr_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function llsr_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.llsr_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{llsr_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function llsr_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.llsr_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{llsr_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export llsr_read_specfile

function llsr_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.llsr_read_specfile_s(control::Ptr{llsr_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function llsr_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.llsr_read_specfile(control::Ptr{llsr_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function llsr_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.llsr_read_specfile_q(control::Ptr{llsr_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export llsr_import

function llsr_import(::Type{Float32}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_single.llsr_import_s(control::Ptr{llsr_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                         n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

function llsr_import(::Type{Float64}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_double.llsr_import(control::Ptr{llsr_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       n::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                       A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                       A_ptr::Ptr{Cint})::Cvoid
end

function llsr_import(::Type{Float128}, control, data, status, m, n, A_type, A_ne, A_row,
                     A_col, A_ptr)
  @ccall libgalahad_quadruple.llsr_import_q(control::Ptr{llsr_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            m::Cint, n::Cint, A_type::Ptr{Cchar},
                                            A_ne::Cint, A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                            A_ptr::Ptr{Cint})::Cvoid
end

export llsr_import_scaling

function llsr_import_scaling(::Type{Float32}, control, data, status, n, S_type, S_ne, S_row,
                             S_col, S_ptr)
  @ccall libgalahad_single.llsr_import_scaling_s(control::Ptr{llsr_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                                 S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                 S_ptr::Ptr{Cint})::Cvoid
end

function llsr_import_scaling(::Type{Float64}, control, data, status, n, S_type, S_ne, S_row,
                             S_col, S_ptr)
  @ccall libgalahad_double.llsr_import_scaling(control::Ptr{llsr_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, S_type::Ptr{Cchar}, S_ne::Cint,
                                               S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                               S_ptr::Ptr{Cint})::Cvoid
end

function llsr_import_scaling(::Type{Float128}, control, data, status, n, S_type, S_ne,
                             S_row, S_col, S_ptr)
  @ccall libgalahad_quadruple.llsr_import_scaling_q(control::Ptr{llsr_control_type{Float128}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    S_type::Ptr{Cchar}, S_ne::Cint,
                                                    S_row::Ptr{Cint}, S_col::Ptr{Cint},
                                                    S_ptr::Ptr{Cint})::Cvoid
end

export llsr_reset_control

function llsr_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.llsr_reset_control_s(control::Ptr{llsr_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function llsr_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.llsr_reset_control(control::Ptr{llsr_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function llsr_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.llsr_reset_control_q(control::Ptr{llsr_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export llsr_solve_problem

function llsr_solve_problem(::Type{Float32}, data, status, m, n, power, weight, A_ne, A_val,
                            b, x, S_ne, S_val)
  @ccall libgalahad_single.llsr_solve_problem_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float32,
                                                weight::Float32, A_ne::Cint,
                                                A_val::Ptr{Float32}, b::Ptr{Float32},
                                                x::Ptr{Float32}, S_ne::Cint,
                                                S_val::Ptr{Float32})::Cvoid
end

function llsr_solve_problem(::Type{Float64}, data, status, m, n, power, weight, A_ne, A_val,
                            b, x, S_ne, S_val)
  @ccall libgalahad_double.llsr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, power::Float64,
                                              weight::Float64, A_ne::Cint,
                                              A_val::Ptr{Float64}, b::Ptr{Float64},
                                              x::Ptr{Float64}, S_ne::Cint,
                                              S_val::Ptr{Float64})::Cvoid
end

function llsr_solve_problem(::Type{Float128}, data, status, m, n, power, weight, A_ne,
                            A_val, b, x, S_ne, S_val)
  @ccall libgalahad_quadruple.llsr_solve_problem_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   m::Cint, n::Cint, power::Float128,
                                                   weight::Float128, A_ne::Cint,
                                                   A_val::Ptr{Float128}, b::Ptr{Float128},
                                                   x::Ptr{Float128}, S_ne::Cint,
                                                   S_val::Ptr{Float128})::Cvoid
end

export llsr_information

function llsr_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.llsr_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{llsr_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function llsr_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.llsr_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{llsr_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function llsr_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.llsr_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{llsr_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export llsr_terminate

function llsr_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.llsr_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{llsr_control_type{Float32}},
                                            inform::Ptr{llsr_inform_type{Float32}})::Cvoid
end

function llsr_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.llsr_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{llsr_control_type{Float64}},
                                          inform::Ptr{llsr_inform_type{Float64}})::Cvoid
end

function llsr_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.llsr_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{llsr_control_type{Float128}},
                                               inform::Ptr{llsr_inform_type{Float128}})::Cvoid
end
