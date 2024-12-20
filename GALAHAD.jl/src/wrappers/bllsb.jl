export bllsb_control_type

struct bllsb_control_type{T}
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
  prfeas::T
  dufeas::T
  muzero::T
  tau::T
  gamma_c::T
  gamma_f::T
  reduce_infeas::T
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
  reduced_pounce_system::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  generate_qplib_file::Bool
  symmetric_linear_solver::NTuple{31,Cchar}
  sif_file_name::NTuple{31,Cchar}
  qplib_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  fdc_control::fdc_control_type{T}
  sls_control::sls_control_type{T}
  sls_pounce_control::sls_control_type{T}
  fit_control::fit_control_type
  roots_control::roots_control_type{T}
  cro_control::cro_control_type{T}
end

export bllsb_time_type

struct bllsb_time_type{T}
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

export bllsb_inform_type

struct bllsb_inform_type{T}
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
  non_negligible_pivot::T
  feasible::Bool
  checkpointsIter::NTuple{16,Cint}
  checkpointsTime::NTuple{16,T}
  time::bllsb_time_type{T}
  fdc_inform::fdc_inform_type{T}
  sls_inform::sls_inform_type{T}
  sls_pounce_inform::sls_inform_type{T}
  fit_inform::fit_inform_type
  roots_inform::roots_inform_type
  cro_inform::cro_inform_type{T}
  rpd_inform::rpd_inform_type
end

export bllsb_initialize

function bllsb_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.bllsb_initialize_s(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{bllsb_control_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function bllsb_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.bllsb_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bllsb_control_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function bllsb_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.bllsb_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{bllsb_control_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export bllsb_read_specfile

function bllsb_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.bllsb_read_specfile_s(control::Ptr{bllsb_control_type{Float32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function bllsb_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.bllsb_read_specfile(control::Ptr{bllsb_control_type{Float64}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function bllsb_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.bllsb_read_specfile_q(control::Ptr{bllsb_control_type{Float128}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export bllsb_import

function bllsb_import(::Type{Float32}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                      Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single.bllsb_import_s(control::Ptr{bllsb_control_type{Float32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                          Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                          Ao_ptr_ne::Cint, Ao_ptr::Ptr{Cint})::Cvoid
end

function bllsb_import(::Type{Float64}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                      Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double.bllsb_import(control::Ptr{bllsb_control_type{Float64}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                        Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                        Ao_ptr_ne::Cint, Ao_ptr::Ptr{Cint})::Cvoid
end

function bllsb_import(::Type{Float128}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                      Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple.bllsb_import_q(control::Ptr{bllsb_control_type{Float128}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, o::Cint, Ao_type::Ptr{Cchar},
                                             Ao_ne::Cint, Ao_row::Ptr{Cint},
                                             Ao_col::Ptr{Cint}, Ao_ptr_ne::Cint,
                                             Ao_ptr::Ptr{Cint})::Cvoid
end

export bllsb_reset_control

function bllsb_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.bllsb_reset_control_s(control::Ptr{bllsb_control_type{Float32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function bllsb_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.bllsb_reset_control(control::Ptr{bllsb_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function bllsb_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.bllsb_reset_control_q(control::Ptr{bllsb_control_type{Float128}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint})::Cvoid
end

export bllsb_solve_blls

function bllsb_solve_blls(::Type{Float32}, data, status, n, o, Ao_ne, Ao_val, b,
                          regularization_weight, x_l, x_u, x, r, z, x_stat, w)
  @ccall libgalahad_single.bllsb_solve_blls_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, o::Cint, Ao_ne::Cint,
                                              Ao_val::Ptr{Float32}, b::Ptr{Float32},
                                              regularization_weight::Float32,
                                              x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                              x::Ptr{Float32}, r::Ptr{Float32},
                                              z::Ptr{Float32}, x_stat::Ptr{Cint},
                                              w::Ptr{Float32})::Cvoid
end

function bllsb_solve_blls(::Type{Float64}, data, status, n, o, Ao_ne, Ao_val, b,
                          regularization_weight, x_l, x_u, x, r, z, x_stat, w)
  @ccall libgalahad_double.bllsb_solve_blls(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, o::Cint, Ao_ne::Cint,
                                            Ao_val::Ptr{Float64}, b::Ptr{Float64},
                                            regularization_weight::Float64,
                                            x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                            x::Ptr{Float64}, r::Ptr{Float64},
                                            z::Ptr{Float64}, x_stat::Ptr{Cint},
                                            w::Ptr{Float64})::Cvoid
end

function bllsb_solve_blls(::Type{Float128}, data, status, n, o, Ao_ne, Ao_val, b,
                          regularization_weight, x_l, x_u, x, r, z, x_stat, w)
  @ccall libgalahad_quadruple.bllsb_solve_blls_q(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, o::Cint, Ao_ne::Cint,
                                                 Ao_val::Ptr{Float128}, b::Ptr{Float128},
                                                 regularization_weight::Cfloat128,
                                                 x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                                 x::Ptr{Float128}, r::Ptr{Float128},
                                                 z::Ptr{Float128}, x_stat::Ptr{Cint},
                                                 w::Ptr{Float128})::Cvoid
end

export bllsb_information

function bllsb_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.bllsb_information_s(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{bllsb_inform_type{Float32}},
                                               status::Ptr{Cint})::Cvoid
end

function bllsb_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.bllsb_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{bllsb_inform_type{Float64}},
                                             status::Ptr{Cint})::Cvoid
end

function bllsb_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.bllsb_information_q(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{bllsb_inform_type{Float128}},
                                                  status::Ptr{Cint})::Cvoid
end

export bllsb_terminate

function bllsb_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.bllsb_terminate_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bllsb_control_type{Float32}},
                                             inform::Ptr{bllsb_inform_type{Float32}})::Cvoid
end

function bllsb_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.bllsb_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bllsb_control_type{Float64}},
                                           inform::Ptr{bllsb_inform_type{Float64}})::Cvoid
end

function bllsb_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.bllsb_terminate_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bllsb_control_type{Float128}},
                                                inform::Ptr{bllsb_inform_type{Float128}})::Cvoid
end
