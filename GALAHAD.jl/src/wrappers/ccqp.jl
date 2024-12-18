export ccqp_control_type

struct ccqp_control_type{T}
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
end

export ccqp_time_type

struct ccqp_time_type{T}
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
end

export ccqp_inform_type

struct ccqp_inform_type{T}
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
  time::ccqp_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sbls_inform::sbls_inform_type{T}
  fit_inform::fit_inform_type
  roots_inform::roots_inform_type
  cro_inform::cro_inform_type{T}
  rpd_inform::rpd_inform_type
end

export ccqp_initialize

function ccqp_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.ccqp_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{ccqp_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function ccqp_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.ccqp_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ccqp_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function ccqp_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.ccqp_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{ccqp_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export ccqp_read_specfile

function ccqp_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.ccqp_read_specfile_s(control::Ptr{ccqp_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function ccqp_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.ccqp_read_specfile(control::Ptr{ccqp_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function ccqp_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.ccqp_read_specfile_q(control::Ptr{ccqp_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export ccqp_import

function ccqp_import(::Type{Float32}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.ccqp_import_s(control::Ptr{ccqp_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                         H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                         H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

function ccqp_import(::Type{Float64}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.ccqp_import(control::Ptr{ccqp_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                       H_row::Ptr{Cint}, H_col::Ptr{Cint}, H_ptr::Ptr{Cint},
                                       A_type::Ptr{Cchar}, A_ne::Cint, A_row::Ptr{Cint},
                                       A_col::Ptr{Cint}, A_ptr::Ptr{Cint})::Cvoid
end

function ccqp_import(::Type{Float128}, control, data, status, n, m, H_type, H_ne, H_row,
                     H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_quadruple.ccqp_import_q(control::Ptr{ccqp_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, H_type::Ptr{Cchar},
                                            H_ne::Cint, H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                            H_ptr::Ptr{Cint}, A_type::Ptr{Cchar},
                                            A_ne::Cint, A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                            A_ptr::Ptr{Cint})::Cvoid
end

export ccqp_reset_control

function ccqp_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.ccqp_reset_control_s(control::Ptr{ccqp_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function ccqp_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.ccqp_reset_control(control::Ptr{ccqp_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function ccqp_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.ccqp_reset_control_q(control::Ptr{ccqp_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export ccqp_solve_qp

function ccqp_solve_qp(::Type{Float32}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                       c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.ccqp_solve_qp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, h_ne::Cint,
                                           H_val::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                           a_ne::Cint, A_val::Ptr{Float32},
                                           c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                           x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                           x::Ptr{Float32}, c::Ptr{Float32},
                                           y::Ptr{Float32}, z::Ptr{Float32},
                                           x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function ccqp_solve_qp(::Type{Float64}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                       c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.ccqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                         g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                         A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                         c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                         x_u::Ptr{Float64}, x::Ptr{Float64},
                                         c::Ptr{Float64}, y::Ptr{Float64}, z::Ptr{Float64},
                                         x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function ccqp_solve_qp(::Type{Float128}, data, status, n, m, h_ne, H_val, g, f, a_ne, A_val,
                       c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.ccqp_solve_qp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, m::Cint, h_ne::Cint,
                                              H_val::Ptr{Float128}, g::Ptr{Float128},
                                              f::Cfloat128, a_ne::Cint,
                                              A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                              c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                              x_u::Ptr{Float128}, x::Ptr{Float128},
                                              c::Ptr{Float128}, y::Ptr{Float128},
                                              z::Ptr{Float128}, x_stat::Ptr{Cint},
                                              c_stat::Ptr{Cint})::Cvoid
end

export ccqp_solve_sldqp

function ccqp_solve_sldqp(::Type{Float32}, data, status, n, m, w, x0, g, f, a_ne, A_val,
                          c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_single.ccqp_solve_sldqp_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, m::Cint, w::Ptr{Float32},
                                              x0::Ptr{Float32}, g::Ptr{Float32}, f::Float32,
                                              a_ne::Cint, A_val::Ptr{Float32},
                                              c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                              x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                              x::Ptr{Float32}, c::Ptr{Float32},
                                              y::Ptr{Float32}, z::Ptr{Float32},
                                              x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function ccqp_solve_sldqp(::Type{Float64}, data, status, n, m, w, x0, g, f, a_ne, A_val,
                          c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_double.ccqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, w::Ptr{Float64},
                                            x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                            a_ne::Cint, A_val::Ptr{Float64},
                                            c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                            x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                            x::Ptr{Float64}, c::Ptr{Float64},
                                            y::Ptr{Float64}, z::Ptr{Float64},
                                            x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function ccqp_solve_sldqp(::Type{Float128}, data, status, n, m, w, x0, g, f, a_ne, A_val,
                          c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
  @ccall libgalahad_quadruple.ccqp_solve_sldqp_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, m::Cint, w::Ptr{Float128},
                                                 x0::Ptr{Float128}, g::Ptr{Float128},
                                                 f::Cfloat128, a_ne::Cint,
                                                 A_val::Ptr{Float128}, c_l::Ptr{Float128},
                                                 c_u::Ptr{Float128}, x_l::Ptr{Float128},
                                                 x_u::Ptr{Float128}, x::Ptr{Float128},
                                                 c::Ptr{Float128}, y::Ptr{Float128},
                                                 z::Ptr{Float128}, x_stat::Ptr{Cint},
                                                 c_stat::Ptr{Cint})::Cvoid
end

export ccqp_information

function ccqp_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.ccqp_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{ccqp_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function ccqp_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.ccqp_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{ccqp_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function ccqp_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.ccqp_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{ccqp_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export ccqp_terminate

function ccqp_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.ccqp_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{ccqp_control_type{Float32}},
                                            inform::Ptr{ccqp_inform_type{Float32}})::Cvoid
end

function ccqp_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.ccqp_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ccqp_control_type{Float64}},
                                          inform::Ptr{ccqp_inform_type{Float64}})::Cvoid
end

function ccqp_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.ccqp_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ccqp_control_type{Float128}},
                                               inform::Ptr{ccqp_inform_type{Float128}})::Cvoid
end
