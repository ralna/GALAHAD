export eqp_control_type

struct eqp_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  factorization::Cint
  max_col::Cint
  indmin::Cint
  valmin::Cint
  len_ulsmin::Cint
  itref_max::Cint
  cg_maxit::Cint
  preconditioner::Cint
  semi_bandwidth::Cint
  new_a::Cint
  new_h::Cint
  sif_file_device::Cint
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
  fdc_control::fdc_control_type{T}
  sbls_control::sbls_control_type{T}
  gltr_control::gltr_control_type{T}
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

struct eqp_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  cg_iter::Cint
  cg_iter_inter::Cint
  factorization_integer::Int64
  factorization_real::Int64
  obj::T
  time::eqp_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  gltr_inform::gltr_inform_type{T}
end

export eqp_initialize

function eqp_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.eqp_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{eqp_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

function eqp_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.eqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{eqp_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

function eqp_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.eqp_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{eqp_control_type{Float128}},
                                               status::Ptr{Cint})::Cvoid
end

export eqp_read_specfile

function eqp_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.eqp_read_specfile_s(control::Ptr{eqp_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.eqp_read_specfile(control::Ptr{eqp_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function eqp_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.eqp_read_specfile_q(control::Ptr{eqp_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export eqp_import

function eqp_import(::Type{Float32}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.eqp_import_s(control::Ptr{eqp_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function eqp_import(::Type{Float64}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.eqp_import(control::Ptr{eqp_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                      H_row::Ptr{Cint}, H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                      A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                      A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function eqp_import(::Type{Float128}, control, data, status, n, m, H_type, H_ne, H_row,
                    H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.eqp_import_q(control::Ptr{eqp_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                           H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                           H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                           A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                           A_ptr::Ptr{Cint})::Cvoid
end

export eqp_reset_control

function eqp_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.eqp_reset_control_s(control::Ptr{eqp_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function eqp_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.eqp_reset_control(control::Ptr{eqp_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function eqp_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.eqp_reset_control_q(control::Ptr{eqp_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export eqp_solve_qp

function eqp_solve_qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c, x, y)
  @ccall libgalahad_single.eqp_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                          g::Ptr{Float32}, f::Float32, a_ne::Cint,
                                          A_val::Ptr{Float32}, c::Ptr{Float32},
                                          x::Ptr{Float32}, y::Ptr{Float32})::Cvoid
end

function eqp_solve_qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c, x, y)
  @ccall libgalahad_double.eqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                        g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                        A_val::Ptr{Float64}, c::Ptr{Float64},
                                        x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function eqp_solve_qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                      c, x, y)
  @ccall libgalahad_quadruple.eqp_solve_qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, h_ne::Cint,
                                             H_val::Ptr{Float128}, g::Ptr{Float128},
                                             f::Cfloat128, a_ne::Cint, A_val::Ptr{Float128},
                                             c::Ptr{Float128}, x::Ptr{Float128},
                                             y::Ptr{Float128})::Cvoid
end

export eqp_solve_sldqp

function eqp_solve_sldqp(::Type{Float32}, data, status, n, m, w, x0, g, f, a_ne, A_val, c,
                         x, y)
  @ccall libgalahad_single.eqp_solve_sldqp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, w::Ptr{Float32},
                                             x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                             a_ne::Cint, A_val::Ptr{Float32},
                                             c::Ptr{Float32}, x::Ptr{Float32},
                                             y::Ptr{Float32})::Cvoid
end

function eqp_solve_sldqp(::Type{Float64}, data, status, n, m, w, x0, g, f, a_ne, A_val, c,
                         x, y)
  @ccall libgalahad_double.eqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, w::Ptr{Float64},
                                           x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Cint, A_val::Ptr{Float64}, c::Ptr{Float64},
                                           x::Ptr{Float64}, y::Ptr{Float64})::Cvoid
end

function eqp_solve_sldqp(::Type{Float128}, data, status, n, m, w, x0, g, f, a_ne, A_val, c,
                         x, y)
  @ccall libgalahad_quadruple.eqp_solve_sldqp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, m::Cint, w::Ptr{Float128},
                                                x0::Ptr{Float128}, g::Ptr{Float128},
                                                f::Cfloat128, a_ne::Cint,
                                                A_val::Ptr{Float128}, c::Ptr{Float128},
                                                x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

export eqp_resolve_qp

function eqp_resolve_qp(::Type{Float32}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_single.eqp_resolve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, g::Ptr{Float32}, f::Float32,
                                            c::Ptr{Float32}, x::Ptr{Float32},
                                            y::Ptr{Float32})::Cvoid
end

function eqp_resolve_qp(::Type{Float64}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_double.eqp_resolve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float64}, f::Float64,
                                          c::Ptr{Float64}, x::Ptr{Float64},
                                          y::Ptr{Float64})::Cvoid
end

function eqp_resolve_qp(::Type{Float128}, data, status, n, m, g, f, c, x, y)
  @ccall libgalahad_quadruple.eqp_resolve_qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, m::Cint, g::Ptr{Float128},
                                               f::Cfloat128, c::Ptr{Float128},
                                               x::Ptr{Float128}, y::Ptr{Float128})::Cvoid
end

export eqp_information

function eqp_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.eqp_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{eqp_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function eqp_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.eqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{eqp_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function eqp_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.eqp_information_q(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{eqp_inform_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export eqp_terminate

function eqp_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.eqp_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{eqp_control_type{Float32}},
                                           inform::Ptr{eqp_inform_type{Float32}})::Cvoid
end

function eqp_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.eqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{eqp_control_type{Float64}},
                                         inform::Ptr{eqp_inform_type{Float64}})::Cvoid
end

function eqp_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.eqp_terminate_q(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{eqp_control_type{Float128}},
                                              inform::Ptr{eqp_inform_type{Float128}})::Cvoid
end
