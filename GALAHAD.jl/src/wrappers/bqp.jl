export bqp_control_type

mutable struct bqp_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  cold_start::Cint
  ratio_cg_vs_sd::Cint
  change_max::Cint
  cg_maxit::Cint
  sif_file_device::Cint
  infinity::T
  stop_p::T
  stop_d::T
  stop_c::T
  identical_bounds_tol::T
  stop_cg_relative::T
  stop_cg_absolute::T
  zero_curvature::T
  cpu_time_limit::T
  exact_arcsearch::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T}

  function bqp_control_type{T}() where T
    type = new()
    type.sbls_control = sbls_control_type{T}()
    return type
  end
end

export bqp_time_type

mutable struct bqp_time_type
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32

  bqp_time_type() = new()
end

export bqp_inform_type

mutable struct bqp_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorization_status::Cint
  iter::Cint
  cg_iter::Cint
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::bqp_time_type
  sbls_inform::sbls_inform_type{T}

  function bqp_inform_type{T}() where T
    type = new()
    type.time = bqp_time_type()
    type.sbls_inform = sbls_inform_type{T}()
    return type
  end
end

export bqp_initialize_s

function bqp_initialize_s(data, control, status)
  @ccall libgalahad_single.bqp_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{bqp_control_type{Float32}},
                                            status::Ptr{Cint})::Cvoid
end

export bqp_initialize

function bqp_initialize(data, control, status)
  @ccall libgalahad_double.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{bqp_control_type{Float64}},
                                          status::Ptr{Cint})::Cvoid
end

export bqp_read_specfile_s

function bqp_read_specfile_s(control, specfile)
  @ccall libgalahad_single.bqp_read_specfile_s(control::Ref{bqp_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export bqp_read_specfile

function bqp_read_specfile(control, specfile)
  @ccall libgalahad_double.bqp_read_specfile(control::Ref{bqp_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export bqp_import_s

function bqp_import_s(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.bqp_import_s(control::Ref{bqp_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bqp_import

function bqp_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.bqp_import(control::Ref{bqp_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                      H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                      H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bqp_import_without_h_s

function bqp_import_without_h_s(control, data, status, n)
  @ccall libgalahad_single.bqp_import_without_h_s(control::Ref{bqp_control_type{Float32}},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint)::Cvoid
end

export bqp_import_without_h

function bqp_import_without_h(control, data, status, n)
  @ccall libgalahad_double.bqp_import_without_h(control::Ref{bqp_control_type{Float64}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint)::Cvoid
end

export bqp_reset_control_s

function bqp_reset_control_s(control, data, status)
  @ccall libgalahad_single.bqp_reset_control_s(control::Ref{bqp_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export bqp_reset_control

function bqp_reset_control(control, data, status)
  @ccall libgalahad_double.bqp_reset_control(control::Ref{bqp_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export bqp_solve_given_h_s

function bqp_solve_given_h_s(data, status, n, h_ne, H_val, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_single.bqp_solve_given_h_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                               g::Ptr{Float32}, f::Float32,
                                               x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                               x::Ptr{Float32}, z::Ptr{Float32},
                                               x_stat::Ptr{Cint})::Cvoid
end

export bqp_solve_given_h

function bqp_solve_given_h(data, status, n, h_ne, H_val, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_double.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                             g::Ptr{Float64}, f::Float64,
                                             x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                             x::Ptr{Float64}, z::Ptr{Float64},
                                             x_stat::Ptr{Cint})::Cvoid
end

export bqp_solve_reverse_h_prod_s

function bqp_solve_reverse_h_prod_s(data, status, n, g, f, x_l, x_u, x, z, x_stat, v, prod,
                                  nz_v, nz_v_start, nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_single.bqp_solve_reverse_h_prod_s(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      g::Ptr{Float32}, f::Float32,
                                                      x_l::Ptr{Float32},
                                                      x_u::Ptr{Float32}, x::Ptr{Float32},
                                                      z::Ptr{Float32}, x_stat::Ptr{Cint},
                                                      v::Ptr{Float32}, prod::Ptr{Float32},
                                                      nz_v::Ptr{Cint},
                                                      nz_v_start::Ptr{Cint},
                                                      nz_v_end::Ptr{Cint},
                                                      nz_prod::Ptr{Cint},
                                                      nz_prod_end::Cint)::Cvoid
end

export bqp_solve_reverse_h_prod

function bqp_solve_reverse_h_prod(data, status, n, g, f, x_l, x_u, x, z, x_stat, v, prod,
                                nz_v, nz_v_start, nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_double.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    g::Ptr{Float64}, f::Float64,
                                                    x_l::Ptr{Float64},
                                                    x_u::Ptr{Float64}, x::Ptr{Float64},
                                                    z::Ptr{Float64}, x_stat::Ptr{Cint},
                                                    v::Ptr{Float64}, prod::Ptr{Float64},
                                                    nz_v::Ptr{Cint},
                                                    nz_v_start::Ptr{Cint},
                                                    nz_v_end::Ptr{Cint},
                                                    nz_prod::Ptr{Cint},
                                                    nz_prod_end::Cint)::Cvoid
end

export bqp_information_s

function bqp_information_s(data, inform, status)
  @ccall libgalahad_single.bqp_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{bqp_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export bqp_information

function bqp_information(data, inform, status)
  @ccall libgalahad_double.bqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{bqp_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export bqp_terminate_s

function bqp_terminate_s(data, control, inform)
  @ccall libgalahad_single.bqp_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{bqp_control_type{Float32}},
                                           inform::Ref{bqp_inform_type{Float32}})::Cvoid
end

export bqp_terminate

function bqp_terminate(data, control, inform)
  @ccall libgalahad_double.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{bqp_control_type{Float64}},
                                         inform::Ref{bqp_inform_type{Float64}})::Cvoid
end
