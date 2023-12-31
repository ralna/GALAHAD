export bqpb_control_type

mutable struct bqpb_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  maxit::Cint
  infeas_max::Cint
  muzero_fixed::Cint
  restore_problem::Cint
  indicator_type::Cint
  arc::Cint
  series_order::Cint
  sif_file_device::Cint
  qplib_file_device::Cint
  infinity::T
  stop_abs_p::T
  stop_rel_p::T
  stop_abs_d::T
  stop_rel_d::T
  stop_abs_c::T
  stop_rel_c::T
  perturb_h::T
  prfeas::T
  dufeas::T
  muzero::T
  tau::T
  gamma_c::T
  gamma_f::T
  reduce_infeas::T
  obj_unbounded::T
  potential_unbounded::T
  identical_bounds_tol::T
  mu_pounce::T
  indicator_tol_p::T
  indicator_tol_pd::T
  indicator_tol_tapia::T
  cpu_time_limit::T
  clock_time_limit::T
  remove_dependencies::Bool
  treat_zero_bounds_as_general::Bool
  treat_separable_as_general::Bool
  just_feasible::Bool
  getdua::Bool
  puiseux::Bool
  every_order::Bool
  feasol::Bool
  balance_initial_complentarity::Bool
  crossover::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  generate_qplib_file::Bool
  sif_file_name::NTuple{31,Cchar}
  qplib_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  fdc_control::fdc_control_type{T}
  sbls_control::sbls_control_type{T}
  fit_control::fit_control_type
  roots_control::roots_control_type{T}
  cro_control::cro_control_type{T}

  function bqpb_control_type{T}() where T
    type = new()
    type.fdc_control = fdc_control_type{T}()
    type.sbls_control = sbls_control_type{T}()
    type.fit_control = fit_control_type()
    type.roots_control = roots_control_type{T}()
    type.cro_control = cro_control_type{T}()
    return type
  end
end

export bqpb_time_type

mutable struct bqpb_time_type{T}
  total::T
  preprocess::T
  find_dependent::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_preprocess::T
  clock_find_dependent::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T

  bqpb_time_type{T}() where T = new()
end

export bqpb_inform_type

mutable struct bqpb_inform_type{T}
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  iter::Cint
  factorization_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::Cint
  nbacts::Cint
  threads::Cint
  obj::T
  primal_infeasibility::T
  dual_infeasibility::T
  complementary_slackness::T
  init_primal_infeasibility::T
  init_dual_infeasibility::T
  init_complementary_slackness::T
  potential::T
  non_negligible_pivot::T
  feasible::Bool
  checkpointsIter::NTuple{16,Cint}
  checkpointsTime::NTuple{16,T}
  time::bqpb_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  fit_inform::fit_inform_type
  roots_inform::roots_inform_type
  cro_inform::cro_inform_type{T}
  rpd_inform::rpd_inform_type

  function bqpb_inform_type{T}() where T
    type = new()
    type.time = bqpb_time_type{T}()
    type.fdc_inform = fdc_inform_type{T}()
    type.sbls_inform = sbls_inform_type{T}()
    type.fit_inform = fit_inform_type()
    type.roots_inform = roots_inform_type()
    type.cro_inform = cro_inform_type{T}()
    type.rpd_inform = rpd_inform_type()
    return type
  end
end

export bqpb_initialize_s

function bqpb_initialize_s(data, control, status)
  @ccall libgalahad_single.bqpb_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{bqpb_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export bqpb_initialize

function bqpb_initialize(data, control, status)
  @ccall libgalahad_double.bqpb_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{bqpb_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export bqpb_read_specfile_s

function bqpb_read_specfile_s(control, specfile)
  @ccall libgalahad_single.bqpb_read_specfile_s(control::Ref{bqpb_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export bqpb_read_specfile

function bqpb_read_specfile(control, specfile)
  @ccall libgalahad_double.bqpb_read_specfile(control::Ref{bqpb_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export bqpb_import_s

function bqpb_import_s(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_single.bqpb_import_s(control::Ref{bqpb_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                         H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bqpb_import

function bqpb_import(control, data, status, n, H_type, H_ne, H_row, H_col, H_ptr)
  @ccall libgalahad_double.bqpb_import(control::Ref{bqpb_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       H_type::Ptr{Cchar}, H_ne::Cint, H_row::Ptr{Cint},
                                       H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

export bqpb_reset_control_s

function bqpb_reset_control_s(control, data, status)
  @ccall libgalahad_single.bqpb_reset_control_s(control::Ref{bqpb_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export bqpb_reset_control

function bqpb_reset_control(control, data, status)
  @ccall libgalahad_double.bqpb_reset_control(control::Ref{bqpb_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export bqpb_solve_qp_s

function bqpb_solve_qp_s(data, status, n, h_ne, H_val, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_single.bqpb_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, h_ne::Cint, H_val::Ptr{Float32},
                                           g::Ptr{Float32}, f::Float32,
                                           x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                           x::Ptr{Float32}, z::Ptr{Float32},
                                           x_stat::Ptr{Cint})::Cvoid
end

export bqpb_solve_qp

function bqpb_solve_qp(data, status, n, h_ne, H_val, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_double.bqpb_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                         n::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                         g::Ptr{Float64}, f::Float64,
                                         x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                         x::Ptr{Float64}, z::Ptr{Float64},
                                         x_stat::Ptr{Cint})::Cvoid
end

export bqpb_solve_sldqp_s

function bqpb_solve_sldqp_s(data, status, n, w, x0, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_single.bqpb_solve_sldqp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, w::Ptr{Float32}, x0::Ptr{Float32},
                                              g::Ptr{Float32}, f::Float32,
                                              x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                              x::Ptr{Float32}, z::Ptr{Float32},
                                              x_stat::Ptr{Cint})::Cvoid
end

export bqpb_solve_sldqp

function bqpb_solve_sldqp(data, status, n, w, x0, g, f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_double.bqpb_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, w::Ptr{Float64}, x0::Ptr{Float64},
                                            g::Ptr{Float64}, f::Float64,
                                            x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                            x::Ptr{Float64}, z::Ptr{Float64},
                                            x_stat::Ptr{Cint})::Cvoid
end

export bqpb_information_s

function bqpb_information_s(data, inform, status)
  @ccall libgalahad_single.bqpb_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{bqpb_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export bqpb_information

function bqpb_information(data, inform, status)
  @ccall libgalahad_double.bqpb_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{bqpb_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export bqpb_terminate_s

function bqpb_terminate_s(data, control, inform)
  @ccall libgalahad_single.bqpb_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{bqpb_control_type{Float32}},
                                            inform::Ref{bqpb_inform_type{Float32}})::Cvoid
end

export bqpb_terminate

function bqpb_terminate(data, control, inform)
  @ccall libgalahad_double.bqpb_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{bqpb_control_type{Float64}},
                                          inform::Ref{bqpb_inform_type{Float64}})::Cvoid
end
