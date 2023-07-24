export cro_control_type

mutable struct cro_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  max_schur_complement::Cint
  infinity::T
  feasibility_tolerance::T
  check_io::Bool
  refine_solution::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  unsymmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
  sbls_control::sbls_control_type{T}
  uls_control::uls_control_type{T}
  ir_control::ir_control_type{T}

  cro_control_type{T}() where T = new()
end

export cro_time_type

mutable struct cro_time_type{T}
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T

  cro_time_type{T}() where T = new()
end

export cro_inform_type

mutable struct cro_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  dependent::Cint
  time::cro_time_type{T}
  sls_inform::sls_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  uls_inform::uls_inform_type{T}
  scu_status::Cint
  scu_inform::scu_inform_type
  ir_inform::ir_inform_type{T}

  cro_inform_type{T}() where T = new()
end

export cro_initialize_s

function cro_initialize_s(data, control, status)
  @ccall libgalahad_single.cro_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{cro_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export cro_initialize

function cro_initialize(data, control, status)
  @ccall libgalahad_double.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{cro_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export cro_read_specfile_s

function cro_read_specfile_s(control, specfile)
  @ccall libgalahad_single.cro_read_specfile_s(control::Ref{cro_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export cro_read_specfile

function cro_read_specfile(control, specfile)
  @ccall libgalahad_double.cro_read_specfile(control::Ref{cro_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export cro_crossover_solution_s

function cro_crossover_solution_s(data, control, inform, n, m, m_equal, h_ne, H_val, H_col,
                                H_ptr, a_ne, A_val, A_col, A_ptr, g, c_l, c_u, x_l, x_u, x,
                                c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.cro_crossover_solution_s(data::Ptr{Ptr{Cvoid}},
                                                    control::Ref{cro_control_type{Float32}},
                                                    inform::Ref{cro_inform_type{Float32}}, n::Cint,
                                                    m::Cint, m_equal::Cint, h_ne::Cint,
                                                    H_val::Ptr{Float32}, H_col::Ptr{Cint},
                                                    H_ptr::Ptr{Cint}, a_ne::Cint,
                                                    A_val::Ptr{Float32}, A_col::Ptr{Cint},
                                                    A_ptr::Ptr{Cint}, g::Ptr{Float32},
                                                    c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                                    x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                    x::Ptr{Float32}, c::Ptr{Float32},
                                                    y::Ptr{Float32}, z::Ptr{Float32},
                                                    x_stat::Ptr{Cint},
                                                    c_stat::Ptr{Cint})::Cvoid
end

export cro_crossover_solution

function cro_crossover_solution(data, control, inform, n, m, m_equal, h_ne, H_val, H_col,
                              H_ptr, a_ne, A_val, A_col, A_ptr, g, c_l, c_u, x_l, x_u, x,
                              c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                  control::Ref{cro_control_type{Float64}},
                                                  inform::Ref{cro_inform_type{Float64}}, n::Cint,
                                                  m::Cint, m_equal::Cint, h_ne::Cint,
                                                  H_val::Ptr{Float64}, H_col::Ptr{Cint},
                                                  H_ptr::Ptr{Cint}, a_ne::Cint,
                                                  A_val::Ptr{Float64}, A_col::Ptr{Cint},
                                                  A_ptr::Ptr{Cint}, g::Ptr{Float64},
                                                  c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                  x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                  x::Ptr{Float64}, c::Ptr{Float64},
                                                  y::Ptr{Float64}, z::Ptr{Float64},
                                                  x_stat::Ptr{Cint},
                                                  c_stat::Ptr{Cint})::Cvoid
end

export cro_terminate_s

function cro_terminate_s(data, control, inform)
  @ccall libgalahad_single.cro_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{cro_control_type{Float32}},
                                           inform::Ref{cro_inform_type{Float32}})::Cvoid
end

export cro_terminate

function cro_terminate(data, control, inform)
  @ccall libgalahad_double.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{cro_control_type{Float64}},
                                         inform::Ref{cro_inform_type{Float64}})::Cvoid
end
