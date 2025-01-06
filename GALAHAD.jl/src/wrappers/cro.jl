export cro_control_type

struct cro_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  max_schur_complement::INT
  infinity::T
  feasibility_tolerance::T
  check_io::Bool
  refine_solution::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  unsymmetric_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
  sbls_control::sbls_control_type{T,INT}
  uls_control::uls_control_type{T,INT}
  ir_control::ir_control_type{T,INT}
end

export cro_time_type

struct cro_time_type{T}
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export cro_inform_type

struct cro_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  dependent::INT
  time::cro_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  sbls_inform::sbls_inform_type{T,INT}
  uls_inform::uls_inform_type{T,INT}
  scu_status::INT
  scu_inform::scu_inform_type{INT}
  ir_inform::ir_inform_type{T,INT}
end

export cro_initialize

function cro_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{cro_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function cro_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{cro_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function cro_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{cro_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function cro_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{cro_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function cro_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{cro_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function cro_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.cro_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{cro_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export cro_read_specfile

function cro_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.cro_read_specfile(control::Ptr{cro_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function cro_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.cro_read_specfile(control::Ptr{cro_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function cro_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.cro_read_specfile(control::Ptr{cro_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function cro_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.cro_read_specfile(control::Ptr{cro_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function cro_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.cro_read_specfile(control::Ptr{cro_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function cro_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.cro_read_specfile(control::Ptr{cro_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export cro_crossover_solution

function cro_crossover_solution(::Type{Float32}, ::Type{Int32}, data, control, inform, n, m,
                                m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{cro_control_type{Float32,
                                                                                Int32}},
                                                  inform::Ptr{cro_inform_type{Float32,
                                                                              Int32}},
                                                  n::Int32, m::Int32, m_equal::Int32,
                                                  h_ne::Int32, H_val::Ptr{Float32},
                                                  H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                  a_ne::Int32, A_val::Ptr{Float32},
                                                  A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                  g::Ptr{Float32}, c_l::Ptr{Float32},
                                                  c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                                  x_u::Ptr{Float32}, x::Ptr{Float32},
                                                  c::Ptr{Float32}, y::Ptr{Float32},
                                                  z::Ptr{Float32}, x_stat::Ptr{Int32},
                                                  c_stat::Ptr{Int32})::Cvoid
end

function cro_crossover_solution(::Type{Float32}, ::Type{Int64}, data, control, inform, n, m,
                                m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single_64.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{cro_control_type{Float32,
                                                                                   Int64}},
                                                     inform::Ptr{cro_inform_type{Float32,
                                                                                 Int64}},
                                                     n::Int64, m::Int64, m_equal::Int64,
                                                     h_ne::Int64, H_val::Ptr{Float32},
                                                     H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                                     a_ne::Int64, A_val::Ptr{Float32},
                                                     A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                     g::Ptr{Float32}, c_l::Ptr{Float32},
                                                     c_u::Ptr{Float32}, x_l::Ptr{Float32},
                                                     x_u::Ptr{Float32}, x::Ptr{Float32},
                                                     c::Ptr{Float32}, y::Ptr{Float32},
                                                     z::Ptr{Float32}, x_stat::Ptr{Int64},
                                                     c_stat::Ptr{Int64})::Cvoid
end

function cro_crossover_solution(::Type{Float64}, ::Type{Int32}, data, control, inform, n, m,
                                m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{cro_control_type{Float64,
                                                                                Int32}},
                                                  inform::Ptr{cro_inform_type{Float64,
                                                                              Int32}},
                                                  n::Int32, m::Int32, m_equal::Int32,
                                                  h_ne::Int32, H_val::Ptr{Float64},
                                                  H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                  a_ne::Int32, A_val::Ptr{Float64},
                                                  A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                  g::Ptr{Float64}, c_l::Ptr{Float64},
                                                  c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                                  x_u::Ptr{Float64}, x::Ptr{Float64},
                                                  c::Ptr{Float64}, y::Ptr{Float64},
                                                  z::Ptr{Float64}, x_stat::Ptr{Int32},
                                                  c_stat::Ptr{Int32})::Cvoid
end

function cro_crossover_solution(::Type{Float64}, ::Type{Int64}, data, control, inform, n, m,
                                m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double_64.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{cro_control_type{Float64,
                                                                                   Int64}},
                                                     inform::Ptr{cro_inform_type{Float64,
                                                                                 Int64}},
                                                     n::Int64, m::Int64, m_equal::Int64,
                                                     h_ne::Int64, H_val::Ptr{Float64},
                                                     H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                                     a_ne::Int64, A_val::Ptr{Float64},
                                                     A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                     g::Ptr{Float64}, c_l::Ptr{Float64},
                                                     c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                                     x_u::Ptr{Float64}, x::Ptr{Float64},
                                                     c::Ptr{Float64}, y::Ptr{Float64},
                                                     z::Ptr{Float64}, x_stat::Ptr{Int64},
                                                     c_stat::Ptr{Int64})::Cvoid
end

function cro_crossover_solution(::Type{Float128}, ::Type{Int32}, data, control, inform, n,
                                m, m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{cro_control_type{Float128,
                                                                                   Int32}},
                                                     inform::Ptr{cro_inform_type{Float128,
                                                                                 Int32}},
                                                     n::Int32, m::Int32, m_equal::Int32,
                                                     h_ne::Int32, H_val::Ptr{Float128},
                                                     H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                     a_ne::Int32, A_val::Ptr{Float128},
                                                     A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                     g::Ptr{Float128}, c_l::Ptr{Float128},
                                                     c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                                     x_u::Ptr{Float128}, x::Ptr{Float128},
                                                     c::Ptr{Float128}, y::Ptr{Float128},
                                                     z::Ptr{Float128}, x_stat::Ptr{Int32},
                                                     c_stat::Ptr{Int32})::Cvoid
end

function cro_crossover_solution(::Type{Float128}, ::Type{Int64}, data, control, inform, n,
                                m, m_equal, h_ne, H_val, H_col, H_ptr, a_ne, A_val, A_col,
                                A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple_64.cro_crossover_solution(data::Ptr{Ptr{Cvoid}},
                                                        control::Ptr{cro_control_type{Float128,
                                                                                      Int64}},
                                                        inform::Ptr{cro_inform_type{Float128,
                                                                                    Int64}},
                                                        n::Int64, m::Int64, m_equal::Int64,
                                                        h_ne::Int64, H_val::Ptr{Float128},
                                                        H_col::Ptr{Int64},
                                                        H_ptr::Ptr{Int64}, a_ne::Int64,
                                                        A_val::Ptr{Float128},
                                                        A_col::Ptr{Int64},
                                                        A_ptr::Ptr{Int64}, g::Ptr{Float128},
                                                        c_l::Ptr{Float128},
                                                        c_u::Ptr{Float128},
                                                        x_l::Ptr{Float128},
                                                        x_u::Ptr{Float128},
                                                        x::Ptr{Float128}, c::Ptr{Float128},
                                                        y::Ptr{Float128}, z::Ptr{Float128},
                                                        x_stat::Ptr{Int64},
                                                        c_stat::Ptr{Int64})::Cvoid
end

export cro_terminate

function cro_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{cro_control_type{Float32,Int32}},
                                         inform::Ptr{cro_inform_type{Float32,Int32}})::Cvoid
end

function cro_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{cro_control_type{Float32,Int64}},
                                            inform::Ptr{cro_inform_type{Float32,Int64}})::Cvoid
end

function cro_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{cro_control_type{Float64,Int32}},
                                         inform::Ptr{cro_inform_type{Float64,Int32}})::Cvoid
end

function cro_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{cro_control_type{Float64,Int64}},
                                            inform::Ptr{cro_inform_type{Float64,Int64}})::Cvoid
end

function cro_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{cro_control_type{Float128,Int32}},
                                            inform::Ptr{cro_inform_type{Float128,Int32}})::Cvoid
end

function cro_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.cro_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{cro_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{cro_inform_type{Float128,Int64}})::Cvoid
end
