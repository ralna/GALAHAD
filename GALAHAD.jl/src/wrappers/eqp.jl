export eqp_control_type

struct eqp_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  factorization::INT
  max_col::INT
  indmin::INT
  valmin::INT
  len_ulsmin::INT
  itref_max::INT
  cg_maxit::INT
  preconditioner::INT
  semi_bandwidth::INT
  new_a::INT
  new_h::INT
  sif_file_device::INT
  pivot_tol::T
  pivot_tol_for_basis::T
  zero_pivot::T
  inner_fraction_opt::T
  radius::T
  min_diagonal::T
  max_infeasibility_relative::T
  max_infeasibility_absolute::T
  inner_stop_relative::T
  inner_stop_absolute::T
  inner_stop_inter::T
  find_basis_by_transpose::Bool
  remove_dependencies::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  fdc_control::fdc_control_type{T,INT}
  sbls_control::sbls_control_type{T,INT}
  gltr_control::gltr_control_type{T,INT}
end

export eqp_time_type

struct eqp_time_type{T}
  total::T
  find_dependent::T
  factorize::T
  solve::T
  solve_inter::T
  clock_total::T
  clock_find_dependent::T
  clock_factorize::T
  clock_solve::T
end

export eqp_inform_type

struct eqp_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  cg_iter::INT
  cg_iter_inter::INT
  factorization_integer::Int64
  factorization_real::Int64
  obj::T
  time::eqp_time_type{T}
  fdc_inform::fdc_inform_type{T,INT}
  sbls_inform::sbls_inform_type{T,INT}
  gltr_inform::gltr_inform_type{T,INT}
end

export eqp_initialize

function eqp_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{eqp_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function eqp_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{eqp_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function eqp_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{eqp_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function eqp_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{eqp_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function eqp_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{eqp_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function eqp_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{eqp_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export eqp_read_specfile

function eqp_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.eqp_read_specfile(control::Ptr{eqp_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.eqp_read_specfile(control::Ptr{eqp_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.eqp_read_specfile(control::Ptr{eqp_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.eqp_read_specfile(control::Ptr{eqp_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.eqp_read_specfile(control::Ptr{eqp_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.eqp_read_specfile(control::Ptr{eqp_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export eqp_import

function eqp_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.eqp_import(control::Ptr{eqp_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                      H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                      H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function eqp_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single_64.eqp_import(control::Ptr{eqp_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, H_type::Ptr{Cchar},
                                         H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64}, A_type::Ptr{Cchar}, A_ne::Int64,
                                         A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function eqp_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.eqp_import(control::Ptr{eqp_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      m::Int32, H_type::Ptr{Cchar}, H_ne::Int32,
                                      H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                      H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                      A_ptr::Ptr{Int32})::Cvoid
end

function eqp_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double_64.eqp_import(control::Ptr{eqp_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, m::Int64, H_type::Ptr{Cchar},
                                         H_ne::Int64, H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64}, A_type::Ptr{Cchar}, A_ne::Int64,
                                         A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                         A_ptr::Ptr{Int64})::Cvoid
end

function eqp_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.eqp_import(control::Ptr{eqp_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, m::Int32, H_type::Ptr{Cchar},
                                         H_ne::Int32, H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32}, A_type::Ptr{Cchar}, A_ne::Int32,
                                         A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                         A_ptr::Ptr{Int32})::Cvoid
end

function eqp_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, H_type,
                    H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple_64.eqp_import(control::Ptr{eqp_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, m::Int64, H_type::Ptr{Cchar},
                                            H_ne::Int64, H_row::Ptr{Int64},
                                            H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                            A_type::Ptr{Cchar}, A_ne::Int64,
                                            A_row::Ptr{Int64}, A_col::Ptr{Int64},
                                            A_ptr::Ptr{Int64})::Cvoid
end

export eqp_reset_control

function eqp_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.eqp_reset_control(control::Ptr{eqp_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function eqp_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.eqp_reset_control(control::Ptr{eqp_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function eqp_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.eqp_reset_control(control::Ptr{eqp_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function eqp_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.eqp_reset_control(control::Ptr{eqp_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function eqp_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.eqp_reset_control(control::Ptr{eqp_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function eqp_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.eqp_reset_control(control::Ptr{eqp_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export eqp_solve_qp

function eqp_solve_qp(::Type{Float32}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c, x, y)
  @ccall libgalahad_single.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, h_ne::Int32, H_val::Ptr{Float32},
                                        g::Ptr{Float32}, f::Float32, a_ne::Int32,
                                        A_val::Ptr{Float32}, c::Ptr{Float32},
                                        x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function eqp_solve_qp(::Type{Float32}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c, x, y)
  @ccall libgalahad_single_64.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, h_ne::Int64,
                                           H_val::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                           a_ne::Int64, A_val::Ptr{Float32},
                                           c::Ptr{Float32}, x::Ptr{Float32},
                                           y::Ptr{Float32})::Cvoid
end

function eqp_solve_qp(::Type{Float64}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c, x, y)
  @ccall libgalahad_double.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        m::Int32, h_ne::Int32, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Int32,
                                        A_val::Ptr{Float64}, c::Ptr{Float64},
                                        x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function eqp_solve_qp(::Type{Float64}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g, f,
                      a_ne, A_val, c, x, y)
  @ccall libgalahad_double_64.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, m::Int64, h_ne::Int64,
                                           H_val::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Int64, A_val::Ptr{Float64},
                                           c::Ptr{Float64}, x::Ptr{Float64},
                                           y::Ptr{Float64})::Cvoid
end

function eqp_solve_qp(::Type{Float128}, ::Type{Int32}, data, status, n, m, h_ne, H_val, g,
                      f, a_ne, A_val, c, x, y)
  @ccall libgalahad_quadruple.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, h_ne::Int32,
                                           H_val::Ptr{Float128}, g::Ptr{Float128},
                                           f::Cfloat128, a_ne::Int32, A_val::Ptr{Float128},
                                           c::Ptr{Float128}, x::Ptr{Float128},
                                           y::Ptr{Float128})::Cvoid
end

function eqp_solve_qp(::Type{Float128}, ::Type{Int64}, data, status, n, m, h_ne, H_val, g,
                      f, a_ne, A_val, c, x, y)
  @ccall libgalahad_quadruple_64.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, h_ne::Int64,
                                              H_val::Ptr{Float128}, g::Ptr{Float128},
                                              f::Cfloat128, a_ne::Int64,
                                              A_val::Ptr{Float128}, c::Ptr{Float128},
                                              x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

export eqp_solve_sldqp

function eqp_solve_sldqp(::Type{Float32}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_single.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, w::Ptr{Float32},
                                           x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                           a_ne::Int32, A_val::Ptr{Float32},
                                           c::Ptr{Float32}, x::Ptr{Float32},
                                           y::Ptr{Float32})::Cvoid
end

function eqp_solve_sldqp(::Type{Float32}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_single_64.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, w::Ptr{Float32},
                                              x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                              a_ne::Int64, A_val::Ptr{Float32},
                                              c::Ptr{Float32}, x::Ptr{Float32},
                                              y::Ptr{Float32})::Cvoid
end

function eqp_solve_sldqp(::Type{Float64}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_double.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                           n::Int32, m::Int32, w::Ptr{Float64},
                                           x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Int32, A_val::Ptr{Float64},
                                           c::Ptr{Float64}, x::Ptr{Float64},
                                           y::Ptr{Float64})::Cvoid
end

function eqp_solve_sldqp(::Type{Float64}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_double_64.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                              n::Int64, m::Int64, w::Ptr{Float64},
                                              x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                              a_ne::Int64, A_val::Ptr{Float64},
                                              c::Ptr{Float64}, x::Ptr{Float64},
                                              y::Ptr{Float64})::Cvoid
end

function eqp_solve_sldqp(::Type{Float128}, ::Type{Int32}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_quadruple.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, m::Int32, w::Ptr{Float128},
                                              x0::Ptr{Float128}, g::Ptr{Float128},
                                              f::Cfloat128, a_ne::Int32,
                                              A_val::Ptr{Float128}, c::Ptr{Float128},
                                              x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

function eqp_solve_sldqp(::Type{Float128}, ::Type{Int64}, data, status, n, m, w, x0, g, f,
                         a_ne, A_val, c, x, y)
  @ccall libgalahad_quadruple_64.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, m::Int64, w::Ptr{Float128},
                                                 x0::Ptr{Float128}, g::Ptr{Float128},
                                                 f::Cfloat128, a_ne::Int64,
                                                 A_val::Ptr{Float128}, c::Ptr{Float128},
                                                 x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

export eqp_resolve_qp

function eqp_resolve_qp(::Type{Float32}, ::Type{Int32}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_single.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, g::Ptr{Float32}, f::Float32,
                                          c::Ptr{Float32}, x::Ptr{Float32},
                                          y::Ptr{Float32})::Cvoid
end

function eqp_resolve_qp(::Type{Float32}, ::Type{Int64}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_single_64.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, g::Ptr{Float32},
                                             f::Float32, c::Ptr{Float32}, x::Ptr{Float32},
                                             y::Ptr{Float32})::Cvoid
end

function eqp_resolve_qp(::Type{Float64}, ::Type{Int32}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_double.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, g::Ptr{Float64}, f::Float64,
                                          c::Ptr{Float64}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function eqp_resolve_qp(::Type{Float64}, ::Type{Int64}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_double_64.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, g::Ptr{Float64},
                                             f::Float64, c::Ptr{Float64}, x::Ptr{Float64},
                                             y::Ptr{Float64})::Cvoid
end

function eqp_resolve_qp(::Type{Float128}, ::Type{Int32}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_quadruple.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, m::Int32, g::Ptr{Float128},
                                             f::Cfloat128, c::Ptr{Float128},
                                             x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

function eqp_resolve_qp(::Type{Float128}, ::Type{Int64}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_quadruple_64.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, m::Int64, g::Ptr{Float128},
                                                f::Cfloat128, c::Ptr{Float128},
                                                x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

export eqp_information

function eqp_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.eqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{eqp_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function eqp_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.eqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{eqp_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function eqp_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.eqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{eqp_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function eqp_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.eqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{eqp_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function eqp_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.eqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{eqp_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function eqp_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.eqp_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{eqp_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export eqp_terminate

function eqp_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{eqp_control_type{Float32,Int32}},
                                         inform::Ptr{eqp_inform_type{Float32,Int32}})::Cvoid
end

function eqp_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{eqp_control_type{Float32,Int64}},
                                            inform::Ptr{eqp_inform_type{Float32,Int64}})::Cvoid
end

function eqp_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{eqp_control_type{Float64,Int32}},
                                         inform::Ptr{eqp_inform_type{Float64,Int32}})::Cvoid
end

function eqp_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{eqp_control_type{Float64,Int64}},
                                            inform::Ptr{eqp_inform_type{Float64,Int64}})::Cvoid
end

function eqp_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{eqp_control_type{Float128,Int32}},
                                            inform::Ptr{eqp_inform_type{Float128,Int32}})::Cvoid
end

function eqp_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{eqp_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{eqp_inform_type{Float128,Int64}})::Cvoid
end
