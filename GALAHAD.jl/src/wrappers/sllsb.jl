export sllsb_control_type

struct sllsb_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  maxit::INT
  infeas_max::INT
  muzero_fixed::INT
  restore_problem::INT
  indicator_type::INT
  arc::INT
  series_order::INT
  sif_file_device::INT
  qplib_file_device::INT
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
  fdc_control::fdc_control_type{T,INT}
  sls_control::sls_control_type{T,INT}
  sls_pounce_control::sls_control_type{T,INT}
  fit_control::fit_control_type{INT}
  roots_control::roots_control_type{T,INT}
  cro_control::cro_control_type{T,INT}
end

export sllsb_time_type

struct sllsb_time_type{T}
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

export sllsb_inform_type

struct sllsb_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  iter::INT
  factorization_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  nfacts::INT
  nbacts::INT
  threads::INT
  obj::T
  ls_obj::T
  primal_infeasibility::T
  dual_infeasibility::T
  complementary_slackness::T
  non_negligible_pivot::T
  feasible::Bool
  checkpointsIter::NTuple{16,INT}
  checkpointsTime::NTuple{16,T}
  time::sllsb_time_type{T}
  fdc_inform::fdc_inform_type{T,INT}
  sls_inform::sls_inform_type{T,INT}
  sls_pounce_inform::sls_inform_type{T,INT}
  fit_inform::fit_inform_type{INT}
  roots_inform::roots_inform_type{INT}
  cro_inform::cro_inform_type{T,INT}
  rpd_inform::rpd_inform_type{INT}
end

export sllsb_initialize

function sllsb_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.sllsb_initialize_s(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{sllsb_control_type{Float32,
                                                                              Int32}},
                                              status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function sllsb_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.sllsb_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{sllsb_control_type{Float32,
                                                                                    Int64}},
                                                    status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function sllsb_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.sllsb_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sllsb_control_type{Float64,
                                                                            Int32}},
                                            status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function sllsb_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.sllsb_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{sllsb_control_type{Float64,
                                                                                  Int64}},
                                                  status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function sllsb_initialize(::Type{Float128}, ::Type{Int32}, data, control,
                          status)
  @ccall libgalahad_quadruple.sllsb_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{sllsb_control_type{Float128,
                                                                                 Int32}},
                                                 status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function sllsb_initialize(::Type{Float128}, ::Type{Int64}, data, control,
                          status)
  @ccall libgalahad_quadruple_64.sllsb_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{sllsb_control_type{Float128,
                                                                                       Int64}},
                                                       status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export sllsb_read_specfile

function sllsb_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.sllsb_read_specfile_s(control::Ptr{sllsb_control_type{Float32,
                                                                                 Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function sllsb_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.sllsb_read_specfile_s_64(control::Ptr{sllsb_control_type{Float32,
                                                                                       Int64}},
                                                       specfile::Ptr{Cchar})::Cvoid
end

function sllsb_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.sllsb_read_specfile(control::Ptr{sllsb_control_type{Float64,
                                                                               Int32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function sllsb_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.sllsb_read_specfile_64(control::Ptr{sllsb_control_type{Float64,
                                                                                     Int64}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function sllsb_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.sllsb_read_specfile_q(control::Ptr{sllsb_control_type{Float128,
                                                                                    Int32}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function sllsb_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.sllsb_read_specfile_q_64(control::Ptr{sllsb_control_type{Float128,
                                                                                          Int64}},
                                                          specfile::Ptr{Cchar})::Cvoid
end

export sllsb_import

function sllsb_import(::Type{Float32}, ::Type{Int32}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_single.sllsb_import_s(control::Ptr{sllsb_control_type{Float32,
                                                                          Int32}},
                                          data::Ptr{Ptr{Cvoid}},
                                          status::Ptr{Int32}, n::Int32,
                                          o::Int32, m::Int32,
                                          Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                          Ao_row::Ptr{Int32},
                                          Ao_col::Ptr{Int32}, Ao_ptr_ne::Int32,
                                          Ao_ptr::Ptr{Int32},
                                          cohort::Ptr{Int32})::Cvoid
end

function sllsb_import(::Type{Float32}, ::Type{Int64}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_single_64.sllsb_import_s_64(control::Ptr{sllsb_control_type{Float32,
                                                                                Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64}, n::Int64,
                                                o::Int64, m::Int64,
                                                Ao_type::Ptr{Cchar},
                                                Ao_ne::Int64,
                                                Ao_row::Ptr{Int64},
                                                Ao_col::Ptr{Int64},
                                                Ao_ptr_ne::Int64,
                                                Ao_ptr::Ptr{Int64},
                                                cohort::Ptr{Int64})::Cvoid
end

function sllsb_import(::Type{Float64}, ::Type{Int32}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_double.sllsb_import(control::Ptr{sllsb_control_type{Float64,
                                                                        Int32}},
                                        data::Ptr{Ptr{Cvoid}},
                                        status::Ptr{Int32}, n::Int32, o::Int32,
                                        m::Int32, Ao_type::Ptr{Cchar},
                                        Ao_ne::Int32, Ao_row::Ptr{Int32},
                                        Ao_col::Ptr{Int32}, Ao_ptr_ne::Int32,
                                        Ao_ptr::Ptr{Int32},
                                        cohort::Ptr{Int32})::Cvoid
end

function sllsb_import(::Type{Float64}, ::Type{Int64}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_double_64.sllsb_import_64(control::Ptr{sllsb_control_type{Float64,
                                                                              Int64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int64}, n::Int64,
                                              o::Int64, m::Int64,
                                              Ao_type::Ptr{Cchar}, Ao_ne::Int64,
                                              Ao_row::Ptr{Int64},
                                              Ao_col::Ptr{Int64},
                                              Ao_ptr_ne::Int64,
                                              Ao_ptr::Ptr{Int64},
                                              cohort::Ptr{Int64})::Cvoid
end

function sllsb_import(::Type{Float128}, ::Type{Int32}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_quadruple.sllsb_import_q(control::Ptr{sllsb_control_type{Float128,
                                                                             Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32}, n::Int32,
                                             o::Int32, m::Int32,
                                             Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                             Ao_row::Ptr{Int32},
                                             Ao_col::Ptr{Int32},
                                             Ao_ptr_ne::Int32,
                                             Ao_ptr::Ptr{Int32},
                                             cohort::Ptr{Int32})::Cvoid
end

function sllsb_import(::Type{Float128}, ::Type{Int64}, control, data, status, n,
                      o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
                      cohort)
  @ccall libgalahad_quadruple_64.sllsb_import_q_64(control::Ptr{sllsb_control_type{Float128,
                                                                                   Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64,
                                                   o::Int64, m::Int64,
                                                   Ao_type::Ptr{Cchar},
                                                   Ao_ne::Int64,
                                                   Ao_row::Ptr{Int64},
                                                   Ao_col::Ptr{Int64},
                                                   Ao_ptr_ne::Int64,
                                                   Ao_ptr::Ptr{Int64},
                                                   cohort::Ptr{Int64})::Cvoid
end

export sllsb_reset_control

function sllsb_reset_control(::Type{Float32}, ::Type{Int32}, control, data,
                             status)
  @ccall libgalahad_single.sllsb_reset_control_s(control::Ptr{sllsb_control_type{Float32,
                                                                                 Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function sllsb_reset_control(::Type{Float32}, ::Type{Int64}, control, data,
                             status)
  @ccall libgalahad_single_64.sllsb_reset_control_s_64(control::Ptr{sllsb_control_type{Float32,
                                                                                       Int64}},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64})::Cvoid
end

function sllsb_reset_control(::Type{Float64}, ::Type{Int32}, control, data,
                             status)
  @ccall libgalahad_double.sllsb_reset_control(control::Ptr{sllsb_control_type{Float64,
                                                                               Int32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32})::Cvoid
end

function sllsb_reset_control(::Type{Float64}, ::Type{Int64}, control, data,
                             status)
  @ccall libgalahad_double_64.sllsb_reset_control_64(control::Ptr{sllsb_control_type{Float64,
                                                                                     Int64}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64})::Cvoid
end

function sllsb_reset_control(::Type{Float128}, ::Type{Int32}, control, data,
                             status)
  @ccall libgalahad_quadruple.sllsb_reset_control_q(control::Ptr{sllsb_control_type{Float128,
                                                                                    Int32}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32})::Cvoid
end

function sllsb_reset_control(::Type{Float128}, ::Type{Int64}, control, data,
                             status)
  @ccall libgalahad_quadruple_64.sllsb_reset_control_q_64(control::Ptr{sllsb_control_type{Float128,
                                                                                          Int64}},
                                                          data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64})::Cvoid
end

export sllsb_solve_given_a

function sllsb_solve_given_a(::Type{Float32}, ::Type{Int32}, data, status, n, o,
                             m, Ao_ne, Ao_val, b, regularization_weight, x, r,
                             y, z, x_stat, w, x_s)
  @ccall libgalahad_single.sllsb_solve_given_a_s(data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32}, n::Int32,
                                                 o::Int32, m::Int32,
                                                 Ao_ne::Int32,
                                                 Ao_val::Ptr{Float32},
                                                 b::Ptr{Float32},
                                                 regularization_weight::Float32,
                                                 x::Ptr{Float32},
                                                 r::Ptr{Float32},
                                                 y::Ptr{Float32},
                                                 z::Ptr{Float32},
                                                 x_stat::Ptr{Int32},
                                                 w::Ptr{Float32},
                                                 x_s::Ptr{Float32})::Cvoid
end

function sllsb_solve_given_a(::Type{Float32}, ::Type{Int64}, data, status, n, o,
                             m, Ao_ne, Ao_val, b, regularization_weight, x, r,
                             y, z, x_stat, w, x_s)
  @ccall libgalahad_single_64.sllsb_solve_given_a_s_64(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64},
                                                       n::Int64, o::Int64,
                                                       m::Int64, Ao_ne::Int64,
                                                       Ao_val::Ptr{Float32},
                                                       b::Ptr{Float32},
                                                       regularization_weight::Float32,
                                                       x::Ptr{Float32},
                                                       r::Ptr{Float32},
                                                       y::Ptr{Float32},
                                                       z::Ptr{Float32},
                                                       x_stat::Ptr{Int64},
                                                       w::Ptr{Float32},
                                                       x_s::Ptr{Float32})::Cvoid
end

function sllsb_solve_given_a(::Type{Float64}, ::Type{Int32}, data, status, n, o,
                             m, Ao_ne, Ao_val, b, regularization_weight, x, r,
                             y, z, x_stat, w, x_s)
  @ccall libgalahad_double.sllsb_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int32}, n::Int32,
                                               o::Int32, m::Int32, Ao_ne::Int32,
                                               Ao_val::Ptr{Float64},
                                               b::Ptr{Float64},
                                               regularization_weight::Float64,
                                               x::Ptr{Float64}, r::Ptr{Float64},
                                               y::Ptr{Float64}, z::Ptr{Float64},
                                               x_stat::Ptr{Int32},
                                               w::Ptr{Float64},
                                               x_s::Ptr{Float64})::Cvoid
end

function sllsb_solve_given_a(::Type{Float64}, ::Type{Int64}, data, status, n, o,
                             m, Ao_ne, Ao_val, b, regularization_weight, x, r,
                             y, z, x_stat, w, x_s)
  @ccall libgalahad_double_64.sllsb_solve_given_a_64(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int64},
                                                     n::Int64, o::Int64,
                                                     m::Int64, Ao_ne::Int64,
                                                     Ao_val::Ptr{Float64},
                                                     b::Ptr{Float64},
                                                     regularization_weight::Float64,
                                                     x::Ptr{Float64},
                                                     r::Ptr{Float64},
                                                     y::Ptr{Float64},
                                                     z::Ptr{Float64},
                                                     x_stat::Ptr{Int64},
                                                     w::Ptr{Float64},
                                                     x_s::Ptr{Float64})::Cvoid
end

function sllsb_solve_given_a(::Type{Float128}, ::Type{Int32}, data, status, n,
                             o, m, Ao_ne, Ao_val, b, regularization_weight, x,
                             r, y, z, x_stat, w, x_s)
  @ccall libgalahad_quadruple.sllsb_solve_given_a_q(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32},
                                                    n::Int32, o::Int32,
                                                    m::Int32, Ao_ne::Int32,
                                                    Ao_val::Ptr{Float128},
                                                    b::Ptr{Float128},
                                                    regularization_weight::Cfloat128,
                                                    x::Ptr{Float128},
                                                    r::Ptr{Float128},
                                                    y::Ptr{Float128},
                                                    z::Ptr{Float128},
                                                    x_stat::Ptr{Int32},
                                                    w::Ptr{Float128},
                                                    x_s::Ptr{Float128})::Cvoid
end

function sllsb_solve_given_a(::Type{Float128}, ::Type{Int64}, data, status, n,
                             o, m, Ao_ne, Ao_val, b, regularization_weight, x,
                             r, y, z, x_stat, w, x_s)
  @ccall libgalahad_quadruple_64.sllsb_solve_given_a_q_64(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64},
                                                          n::Int64, o::Int64,
                                                          m::Int64,
                                                          Ao_ne::Int64,
                                                          Ao_val::Ptr{Float128},
                                                          b::Ptr{Float128},
                                                          regularization_weight::Cfloat128,
                                                          x::Ptr{Float128},
                                                          r::Ptr{Float128},
                                                          y::Ptr{Float128},
                                                          z::Ptr{Float128},
                                                          x_stat::Ptr{Int64},
                                                          w::Ptr{Float128},
                                                          x_s::Ptr{Float128})::Cvoid
end

export sllsb_information

function sllsb_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.sllsb_information_s(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{sllsb_inform_type{Float32,
                                                                             Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function sllsb_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.sllsb_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                     inform::Ptr{sllsb_inform_type{Float32,
                                                                                   Int64}},
                                                     status::Ptr{Int64})::Cvoid
end

function sllsb_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.sllsb_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{sllsb_inform_type{Float64,
                                                                           Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function sllsb_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.sllsb_information_64(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{sllsb_inform_type{Float64,
                                                                                 Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function sllsb_information(::Type{Float128}, ::Type{Int32}, data, inform,
                           status)
  @ccall libgalahad_quadruple.sllsb_information_q(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{sllsb_inform_type{Float128,
                                                                                Int32}},
                                                  status::Ptr{Int32})::Cvoid
end

function sllsb_information(::Type{Float128}, ::Type{Int64}, data, inform,
                           status)
  @ccall libgalahad_quadruple_64.sllsb_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                        inform::Ptr{sllsb_inform_type{Float128,
                                                                                      Int64}},
                                                        status::Ptr{Int64})::Cvoid
end

export sllsb_terminate

function sllsb_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.sllsb_terminate_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sllsb_control_type{Float32,
                                                                             Int32}},
                                             inform::Ptr{sllsb_inform_type{Float32,
                                                                           Int32}})::Cvoid
end

function sllsb_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.sllsb_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{sllsb_control_type{Float32,
                                                                                   Int64}},
                                                   inform::Ptr{sllsb_inform_type{Float32,
                                                                                 Int64}})::Cvoid
end

function sllsb_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.sllsb_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sllsb_control_type{Float64,
                                                                           Int32}},
                                           inform::Ptr{sllsb_inform_type{Float64,
                                                                         Int32}})::Cvoid
end

function sllsb_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.sllsb_terminate_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{sllsb_control_type{Float64,
                                                                                 Int64}},
                                                 inform::Ptr{sllsb_inform_type{Float64,
                                                                               Int64}})::Cvoid
end

function sllsb_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.sllsb_terminate_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sllsb_control_type{Float128,
                                                                                Int32}},
                                                inform::Ptr{sllsb_inform_type{Float128,
                                                                              Int32}})::Cvoid
end

function sllsb_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.sllsb_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{sllsb_control_type{Float128,
                                                                                      Int64}},
                                                      inform::Ptr{sllsb_inform_type{Float128,
                                                                                    Int64}})::Cvoid
end

function run_sif(::Val{:sllsb}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsllsb_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:sllsb}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsllsb_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
