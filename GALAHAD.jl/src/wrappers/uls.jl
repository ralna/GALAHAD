export uls_control_type

mutable struct uls_control_type{T}
  f_indexing::Bool
  error::Cint
  warning::Cint
  out::Cint
  print_level::Cint
  print_level_solver::Cint
  initial_fill_in_factor::Cint
  min_real_factor_size::Cint
  min_integer_factor_size::Cint
  max_factor_size::Int64
  blas_block_size_factorize::Cint
  blas_block_size_solve::Cint
  pivot_control::Cint
  pivot_search_limit::Cint
  minimum_size_for_btf::Cint
  max_iterative_refinements::Cint
  stop_if_singular::Bool
  array_increase_factor::T
  switch_to_full_code_density::T
  array_decrease_factor::T
  relative_pivot_tolerance::T
  absolute_pivot_tolerance::T
  zero_tolerance::T
  acceptable_residual_relative::T
  acceptable_residual_absolute::T
  prefix::NTuple{31,Cchar}

  uls_control_type{T}() where T = new()
end

export uls_inform_type

mutable struct uls_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  more_info::Cint
  out_of_range::Int64
  duplicates::Int64
  entries_dropped::Int64
  workspace_factors::Int64
  compresses::Cint
  entries_in_factors::Int64
  rank::Cint
  structural_rank::Cint
  pivot_control::Cint
  iterative_refinements::Cint
  alternative::Bool
  solver::NTuple{21,Cchar}
  gls_ainfo_type::gls_ainfo_type{T}
  gls_finfo_type::gls_finfo_type{T}
  gls_sinfo_type::gls_sinfo_type
  ma48_ainfo::ma48_ainfo{T}
  ma48_finfo::ma48_finfo{T}
  ma48_sinfo::ma48_sinfo
  lapack_error::Cint

  uls_inform_type{T}() where T = new()
end

export uls_initialize_s

function uls_initialize_s(solver, data, control, status)
  @ccall libgalahad_single.uls_initialize_s(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                            control::Ref{uls_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export uls_initialize

function uls_initialize(solver, data, control, status)
  @ccall libgalahad_double.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ref{uls_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export uls_read_specfile_s

function uls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.uls_read_specfile_s(control::Ref{uls_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export uls_read_specfile

function uls_read_specfile(control, specfile)
  @ccall libgalahad_double.uls_read_specfile(control::Ref{uls_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export uls_factorize_matrix_s

function uls_factorize_matrix_s(control, data, status, m, n, type, ne, val, row, col, ptr)
  @ccall libgalahad_single.uls_factorize_matrix_s(control::Ref{uls_control_type{Float32}},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  m::Cint, n::Cint, type::Ptr{Cchar},
                                                  ne::Cint, val::Ptr{Float32},
                                                  row::Ptr{Cint}, col::Ptr{Cint},
                                                  ptr::Ptr{Cint})::Cvoid
end

export uls_factorize_matrix

function uls_factorize_matrix(control, data, status, m, n, type, ne, val, row, col, ptr)
  @ccall libgalahad_double.uls_factorize_matrix(control::Ref{uls_control_type{Float64}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, type::Ptr{Cchar},
                                                ne::Cint, val::Ptr{Float64},
                                                row::Ptr{Cint}, col::Ptr{Cint},
                                                ptr::Ptr{Cint})::Cvoid
end

export uls_reset_control_s

function uls_reset_control_s(control, data, status)
  @ccall libgalahad_single.uls_reset_control_s(control::Ref{uls_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export uls_reset_control

function uls_reset_control(control, data, status)
  @ccall libgalahad_double.uls_reset_control(control::Ref{uls_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export uls_solve_system_s

function uls_solve_system_s(data, status, m, n, sol, trans)
  @ccall libgalahad_single.uls_solve_system_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              m::Cint, n::Cint, sol::Ptr{Float32},
                                              trans::Bool)::Cvoid
end

export uls_solve_system

function uls_solve_system(data, status, m, n, sol, trans)
  @ccall libgalahad_double.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            m::Cint, n::Cint, sol::Ptr{Float64},
                                            trans::Bool)::Cvoid
end

export uls_information_s

function uls_information_s(data, inform, status)
  @ccall libgalahad_single.uls_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{uls_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export uls_information

function uls_information(data, inform, status)
  @ccall libgalahad_double.uls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{uls_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export uls_terminate_s

function uls_terminate_s(data, control, inform)
  @ccall libgalahad_single.uls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{uls_control_type{Float32}},
                                           inform::Ref{uls_inform_type{Float32}})::Cvoid
end

export uls_terminate

function uls_terminate(data, control, inform)
  @ccall libgalahad_double.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{uls_control_type{Float64}},
                                         inform::Ref{uls_inform_type{Float64}})::Cvoid
end
