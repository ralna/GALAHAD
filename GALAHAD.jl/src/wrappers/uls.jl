export uls_control_type

struct uls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  warning::INT
  out::INT
  print_level::INT
  print_level_solver::INT
  initial_fill_in_factor::INT
  min_real_factor_size::INT
  min_integer_factor_size::INT
  max_factor_size::Int64
  blas_block_size_factorize::INT
  blas_block_size_solve::INT
  pivot_control::INT
  pivot_search_limit::INT
  minimum_size_for_btf::INT
  max_iterative_refinements::INT
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
end

export uls_inform_type

struct uls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  more_info::INT
  out_of_range::Int64
  duplicates::Int64
  entries_dropped::Int64
  workspace_factors::Int64
  compresses::INT
  entries_in_factors::Int64
  rank::INT
  structural_rank::INT
  pivot_control::INT
  iterative_refinements::INT
  alternative::Bool
  solver::NTuple{21,Cchar}
  gls_ainfo::gls_ainfo_type{T,INT}
  gls_finfo::gls_finfo_type{T,INT}
  gls_sinfo::gls_sinfo_type{INT}
  ma48_ainfo::ma48_ainfo{T,INT}
  ma48_finfo::ma48_finfo{T,INT}
  ma48_sinfo::ma48_sinfo{INT}
  lapack_error::INT
end

export uls_initialize

function uls_initialize(::Type{Float32}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_single.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{uls_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function uls_initialize(::Type{Float32}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_single_64.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{uls_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function uls_initialize(::Type{Float64}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_double.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{uls_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function uls_initialize(::Type{Float64}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_double_64.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{uls_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function uls_initialize(::Type{Float128}, ::Type{Int32}, solver, data, control, status)
  @ccall libgalahad_quadruple.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{uls_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function uls_initialize(::Type{Float128}, ::Type{Int64}, solver, data, control, status)
  @ccall libgalahad_quadruple_64.uls_initialize(solver::Ptr{Cchar}, data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{uls_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export uls_read_specfile

function uls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.uls_read_specfile(control::Ptr{uls_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function uls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.uls_read_specfile(control::Ptr{uls_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function uls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.uls_read_specfile(control::Ptr{uls_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function uls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.uls_read_specfile(control::Ptr{uls_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function uls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.uls_read_specfile(control::Ptr{uls_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function uls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.uls_read_specfile(control::Ptr{uls_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export uls_factorize_matrix

function uls_factorize_matrix(::Type{Float32}, ::Type{Int32}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_single.uls_factorize_matrix(control::Ptr{uls_control_type{Float32,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                m::Int32, n::Int32, type::Ptr{Cchar},
                                                ne::Int32, val::Ptr{Float32},
                                                row::Ptr{Int32}, col::Ptr{Int32},
                                                ptr::Ptr{Int32})::Cvoid
end

function uls_factorize_matrix(::Type{Float32}, ::Type{Int64}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_single_64.uls_factorize_matrix(control::Ptr{uls_control_type{Float32,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, m::Int64, n::Int64,
                                                   type::Ptr{Cchar}, ne::Int64,
                                                   val::Ptr{Float32}, row::Ptr{Int64},
                                                   col::Ptr{Int64}, ptr::Ptr{Int64})::Cvoid
end

function uls_factorize_matrix(::Type{Float64}, ::Type{Int32}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_double.uls_factorize_matrix(control::Ptr{uls_control_type{Float64,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                m::Int32, n::Int32, type::Ptr{Cchar},
                                                ne::Int32, val::Ptr{Float64},
                                                row::Ptr{Int32}, col::Ptr{Int32},
                                                ptr::Ptr{Int32})::Cvoid
end

function uls_factorize_matrix(::Type{Float64}, ::Type{Int64}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_double_64.uls_factorize_matrix(control::Ptr{uls_control_type{Float64,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, m::Int64, n::Int64,
                                                   type::Ptr{Cchar}, ne::Int64,
                                                   val::Ptr{Float64}, row::Ptr{Int64},
                                                   col::Ptr{Int64}, ptr::Ptr{Int64})::Cvoid
end

function uls_factorize_matrix(::Type{Float128}, ::Type{Int32}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_quadruple.uls_factorize_matrix(control::Ptr{uls_control_type{Float128,
                                                                                 Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, m::Int32, n::Int32,
                                                   type::Ptr{Cchar}, ne::Int32,
                                                   val::Ptr{Float128}, row::Ptr{Int32},
                                                   col::Ptr{Int32}, ptr::Ptr{Int32})::Cvoid
end

function uls_factorize_matrix(::Type{Float128}, ::Type{Int64}, control, data, status, m, n,
                              type, ne, val, row, col, ptr)
  @ccall libgalahad_quadruple_64.uls_factorize_matrix(control::Ptr{uls_control_type{Float128,
                                                                                    Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, m::Int64,
                                                      n::Int64, type::Ptr{Cchar}, ne::Int64,
                                                      val::Ptr{Float128}, row::Ptr{Int64},
                                                      col::Ptr{Int64},
                                                      ptr::Ptr{Int64})::Cvoid
end

export uls_reset_control

function uls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.uls_reset_control(control::Ptr{uls_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function uls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.uls_reset_control(control::Ptr{uls_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function uls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.uls_reset_control(control::Ptr{uls_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function uls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.uls_reset_control(control::Ptr{uls_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function uls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.uls_reset_control(control::Ptr{uls_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function uls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.uls_reset_control(control::Ptr{uls_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export uls_solve_system

function uls_solve_system(::Type{Float32}, ::Type{Int32}, data, status, m, n, sol, trans)
  @ccall libgalahad_single.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            m::Int32, n::Int32, sol::Ptr{Float32},
                                            trans::Bool)::Cvoid
end

function uls_solve_system(::Type{Float32}, ::Type{Int64}, data, status, m, n, sol, trans)
  @ccall libgalahad_single_64.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               m::Int64, n::Int64, sol::Ptr{Float32},
                                               trans::Bool)::Cvoid
end

function uls_solve_system(::Type{Float64}, ::Type{Int32}, data, status, m, n, sol, trans)
  @ccall libgalahad_double.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            m::Int32, n::Int32, sol::Ptr{Float64},
                                            trans::Bool)::Cvoid
end

function uls_solve_system(::Type{Float64}, ::Type{Int64}, data, status, m, n, sol, trans)
  @ccall libgalahad_double_64.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               m::Int64, n::Int64, sol::Ptr{Float64},
                                               trans::Bool)::Cvoid
end

function uls_solve_system(::Type{Float128}, ::Type{Int32}, data, status, m, n, sol, trans)
  @ccall libgalahad_quadruple.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                               m::Int32, n::Int32, sol::Ptr{Float128},
                                               trans::Bool)::Cvoid
end

function uls_solve_system(::Type{Float128}, ::Type{Int64}, data, status, m, n, sol, trans)
  @ccall libgalahad_quadruple_64.uls_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                  m::Int64, n::Int64, sol::Ptr{Float128},
                                                  trans::Bool)::Cvoid
end

export uls_information

function uls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.uls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{uls_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function uls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.uls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{uls_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function uls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.uls_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{uls_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function uls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.uls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{uls_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function uls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.uls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{uls_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function uls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.uls_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{uls_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export uls_terminate

function uls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{uls_control_type{Float32,Int32}},
                                         inform::Ptr{uls_inform_type{Float32,Int32}})::Cvoid
end

function uls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{uls_control_type{Float32,Int64}},
                                            inform::Ptr{uls_inform_type{Float32,Int64}})::Cvoid
end

function uls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{uls_control_type{Float64,Int32}},
                                         inform::Ptr{uls_inform_type{Float64,Int32}})::Cvoid
end

function uls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{uls_control_type{Float64,Int64}},
                                            inform::Ptr{uls_inform_type{Float64,Int64}})::Cvoid
end

function uls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{uls_control_type{Float128,Int32}},
                                            inform::Ptr{uls_inform_type{Float128,Int32}})::Cvoid
end

function uls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.uls_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{uls_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{uls_inform_type{Float128,Int64}})::Cvoid
end
