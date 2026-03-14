export snls_control_type

struct snls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  alive_unit::INT
  alive_file::NTuple{31,Cchar}
  jacobian_available::INT
  subproblem_solver::INT
  non_monotone::INT
  weight_update_strategy::INT
  stop_r_absolute::T
  stop_r_relative::T
  stop_pg_absolute::T
  stop_pg_relative::T
  stop_s::T
  stop_pg_switch::T
  initial_weight::T
  minimum_weight::T
  eta_successful::T
  eta_very_successful::T
  eta_too_successful::T
  weight_decrease_min::T
  weight_decrease::T
  weight_increase::T
  weight_increase_max::T
  switch_to_newton::T
  cpu_time_limit::T
  clock_time_limit::T
  newton_acceleration::Bool
  magic_step::Bool
  print_obj::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
  slls_control::slls_control_type{T,INT}
  sllsb_control::sllsb_control_type{T,INT}
end

export snls_time_type

struct snls_time_type{T}
  total::T
  slls::T
  sllsb::T
  clock_total::T
  clock_slls::T
  clock_sllsb::T
end

export snls_inform_type

struct snls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  bad_eval::NTuple{13,Cchar}
  iter::INT
  inner_iter::INT
  r_eval::INT
  jr_eval::INT
  obj::T
  norm_r::T
  norm_g::T
  norm_pg::T
  weight::T
  time::snls_time_type{T}
  slls_inform::slls_inform_type{T,INT}
  sllsb_inform::sllsb_inform_type{T,INT}
  lapack_error::INT
end

export snls_initialize

function snls_initialize(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.snls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{snls_control_type{Float32,
                                                                            Int32}},
                                             inform::Ptr{snls_inform_type{Float32,
                                                                          Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function snls_initialize(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.snls_initialize_s_64(data::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{snls_control_type{Float32,
                                                                                  Int64}},
                                                   inform::Ptr{snls_inform_type{Float32,
                                                                                Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function snls_initialize(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.snls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{snls_control_type{Float64,
                                                                          Int32}},
                                           inform::Ptr{snls_inform_type{Float64,
                                                                        Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function snls_initialize(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.snls_initialize_64(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{snls_control_type{Float64,
                                                                                Int64}},
                                                 inform::Ptr{snls_inform_type{Float64,
                                                                              Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function snls_initialize(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.snls_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{snls_control_type{Float128,
                                                                               Int32}},
                                                inform::Ptr{snls_inform_type{Float128,
                                                                             Int32}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function snls_initialize(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.snls_initialize_q_64(data::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{snls_control_type{Float128,
                                                                                     Int64}},
                                                      inform::Ptr{snls_inform_type{Float128,
                                                                                   Int64}})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export snls_read_specfile

function snls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.snls_read_specfile_s(control::Ptr{snls_control_type{Float32,
                                                                               Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function snls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.snls_read_specfile_s_64(control::Ptr{snls_control_type{Float32,
                                                                                     Int64}},
                                                      specfile::Ptr{Cchar})::Cvoid
end

function snls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.snls_read_specfile(control::Ptr{snls_control_type{Float64,
                                                                             Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function snls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.snls_read_specfile_64(control::Ptr{snls_control_type{Float64,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function snls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.snls_read_specfile_q(control::Ptr{snls_control_type{Float128,
                                                                                  Int32}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

function snls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.snls_read_specfile_q_64(control::Ptr{snls_control_type{Float128,
                                                                                        Int64}},
                                                         specfile::Ptr{Cchar})::Cvoid
end

export snls_import

function snls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_single.snls_import_s(control::Ptr{snls_control_type{Float32,
                                                                        Int32}},
                                         data::Ptr{Ptr{Cvoid}},
                                         status::Ptr{Int32}, n::Int32,
                                         m_r::Int32, m_c::Int32,
                                         Jr_type::Ptr{Cchar}, Jr_ne::Int32,
                                         Jr_row::Ptr{Int32}, Jr_col::Ptr{Int32},
                                         Jr_ptr_ne::Int32, Jr_ptr::Ptr{Int32},
                                         cohort::Ptr{Int32})::Cvoid
end

function snls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_single_64.snls_import_s_64(control::Ptr{snls_control_type{Float32,
                                                                              Int64}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Int64}, n::Int64,
                                               m_r::Int64, m_c::Int64,
                                               Jr_type::Ptr{Cchar},
                                               Jr_ne::Int64, Jr_row::Ptr{Int64},
                                               Jr_col::Ptr{Int64},
                                               Jr_ptr_ne::Int64,
                                               Jr_ptr::Ptr{Int64},
                                               cohort::Ptr{Int64})::Cvoid
end

function snls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_double.snls_import(control::Ptr{snls_control_type{Float64,
                                                                      Int32}},
                                       data::Ptr{Ptr{Cvoid}},
                                       status::Ptr{Int32}, n::Int32, m_r::Int32,
                                       m_c::Int32, Jr_type::Ptr{Cchar},
                                       Jr_ne::Int32, Jr_row::Ptr{Int32},
                                       Jr_col::Ptr{Int32}, Jr_ptr_ne::Int32,
                                       Jr_ptr::Ptr{Int32},
                                       cohort::Ptr{Int32})::Cvoid
end

function snls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_double_64.snls_import_64(control::Ptr{snls_control_type{Float64,
                                                                            Int64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int64}, n::Int64,
                                             m_r::Int64, m_c::Int64,
                                             Jr_type::Ptr{Cchar}, Jr_ne::Int64,
                                             Jr_row::Ptr{Int64},
                                             Jr_col::Ptr{Int64},
                                             Jr_ptr_ne::Int64,
                                             Jr_ptr::Ptr{Int64},
                                             cohort::Ptr{Int64})::Cvoid
end

function snls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_quadruple.snls_import_q(control::Ptr{snls_control_type{Float128,
                                                                           Int32}},
                                            data::Ptr{Ptr{Cvoid}},
                                            status::Ptr{Int32}, n::Int32,
                                            m_r::Int32, m_c::Int32,
                                            Jr_type::Ptr{Cchar}, Jr_ne::Int32,
                                            Jr_row::Ptr{Int32},
                                            Jr_col::Ptr{Int32},
                                            Jr_ptr_ne::Int32,
                                            Jr_ptr::Ptr{Int32},
                                            cohort::Ptr{Int32})::Cvoid
end

function snls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n,
                     m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, Jr_ptr_ne,
                     Jr_ptr, cohort)
  @ccall libgalahad_quadruple_64.snls_import_q_64(control::Ptr{snls_control_type{Float128,
                                                                                 Int64}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Int64}, n::Int64,
                                                  m_r::Int64, m_c::Int64,
                                                  Jr_type::Ptr{Cchar},
                                                  Jr_ne::Int64,
                                                  Jr_row::Ptr{Int64},
                                                  Jr_col::Ptr{Int64},
                                                  Jr_ptr_ne::Int64,
                                                  Jr_ptr::Ptr{Int64},
                                                  cohort::Ptr{Int64})::Cvoid
end

export snls_import_without_jac

function snls_import_without_jac(::Type{Float32}, ::Type{Int32}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_single.snls_import_without_jac_s(control::Ptr{snls_control_type{Float32,
                                                                                    Int32}},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32},
                                                     n::Int32, m_r::Int32,
                                                     m_c::Int32,
                                                     cohort::Ptr{Int32})::Cvoid
end

function snls_import_without_jac(::Type{Float32}, ::Type{Int64}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_single_64.snls_import_without_jac_s_64(control::Ptr{snls_control_type{Float32,
                                                                                          Int64}},
                                                           data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int64},
                                                           n::Int64, m_r::Int64,
                                                           m_c::Int64,
                                                           cohort::Ptr{Int64})::Cvoid
end

function snls_import_without_jac(::Type{Float64}, ::Type{Int32}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_double.snls_import_without_jac(control::Ptr{snls_control_type{Float64,
                                                                                  Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32,
                                                   m_r::Int32, m_c::Int32,
                                                   cohort::Ptr{Int32})::Cvoid
end

function snls_import_without_jac(::Type{Float64}, ::Type{Int64}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_double_64.snls_import_without_jac_64(control::Ptr{snls_control_type{Float64,
                                                                                        Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64},
                                                         n::Int64, m_r::Int64,
                                                         m_c::Int64,
                                                         cohort::Ptr{Int64})::Cvoid
end

function snls_import_without_jac(::Type{Float128}, ::Type{Int32}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_quadruple.snls_import_without_jac_q(control::Ptr{snls_control_type{Float128,
                                                                                       Int32}},
                                                        data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32},
                                                        n::Int32, m_r::Int32,
                                                        m_c::Int32,
                                                        cohort::Ptr{Int32})::Cvoid
end

function snls_import_without_jac(::Type{Float128}, ::Type{Int64}, control, data,
                                 status, n, m_r, m_c, cohort)
  @ccall libgalahad_quadruple_64.snls_import_without_jac_q_64(control::Ptr{snls_control_type{Float128,
                                                                                             Int64}},
                                                              data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int64},
                                                              n::Int64,
                                                              m_r::Int64,
                                                              m_c::Int64,
                                                              cohort::Ptr{Int64})::Cvoid
end

export snls_reset_control

function snls_reset_control(::Type{Float32}, ::Type{Int32}, control, data,
                            status)
  @ccall libgalahad_single.snls_reset_control_s(control::Ptr{snls_control_type{Float32,
                                                                               Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function snls_reset_control(::Type{Float32}, ::Type{Int64}, control, data,
                            status)
  @ccall libgalahad_single_64.snls_reset_control_s_64(control::Ptr{snls_control_type{Float32,
                                                                                     Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64})::Cvoid
end

function snls_reset_control(::Type{Float64}, ::Type{Int32}, control, data,
                            status)
  @ccall libgalahad_double.snls_reset_control(control::Ptr{snls_control_type{Float64,
                                                                             Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function snls_reset_control(::Type{Float64}, ::Type{Int64}, control, data,
                            status)
  @ccall libgalahad_double_64.snls_reset_control_64(control::Ptr{snls_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

function snls_reset_control(::Type{Float128}, ::Type{Int32}, control, data,
                            status)
  @ccall libgalahad_quadruple.snls_reset_control_q(control::Ptr{snls_control_type{Float128,
                                                                                  Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32})::Cvoid
end

function snls_reset_control(::Type{Float128}, ::Type{Int64}, control, data,
                            status)
  @ccall libgalahad_quadruple_64.snls_reset_control_q_64(control::Ptr{snls_control_type{Float128,
                                                                                        Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64})::Cvoid
end

export snls_solve_with_jac

function snls_solve_with_jac(::Type{Float32}, ::Type{Int32}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_single.snls_solve_with_jac_s(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid},
                                                 status::Ptr{Int32}, n::Int32,
                                                 m_r::Int32, m_c::Int32,
                                                 x::Ptr{Float32},
                                                 y::Ptr{Float32},
                                                 z::Ptr{Float32},
                                                 r::Ptr{Float32},
                                                 g::Ptr{Float32},
                                                 x_stat::Ptr{Int32},
                                                 eval_r::Ptr{Cvoid},
                                                 jr_ne::Int32,
                                                 eval_jr::Ptr{Cvoid},
                                                 w::Ptr{Float32})::Cvoid
end

function snls_solve_with_jac(::Type{Float32}, ::Type{Int64}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_single_64.snls_solve_with_jac_s_64(data::Ptr{Ptr{Cvoid}},
                                                       userdata::Ptr{Cvoid},
                                                       status::Ptr{Int64},
                                                       n::Int64, m_r::Int64,
                                                       m_c::Int64,
                                                       x::Ptr{Float32},
                                                       y::Ptr{Float32},
                                                       z::Ptr{Float32},
                                                       r::Ptr{Float32},
                                                       g::Ptr{Float32},
                                                       x_stat::Ptr{Int64},
                                                       eval_r::Ptr{Cvoid},
                                                       jr_ne::Int64,
                                                       eval_jr::Ptr{Cvoid},
                                                       w::Ptr{Float32})::Cvoid
end

function snls_solve_with_jac(::Type{Float64}, ::Type{Int32}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_double.snls_solve_with_jac(data::Ptr{Ptr{Cvoid}},
                                               userdata::Ptr{Cvoid},
                                               status::Ptr{Int32}, n::Int32,
                                               m_r::Int32, m_c::Int32,
                                               x::Ptr{Float64}, y::Ptr{Float64},
                                               z::Ptr{Float64}, r::Ptr{Float64},
                                               g::Ptr{Float64},
                                               x_stat::Ptr{Int32},
                                               eval_r::Ptr{Cvoid}, jr_ne::Int32,
                                               eval_jr::Ptr{Cvoid},
                                               w::Ptr{Float64})::Cvoid
end

function snls_solve_with_jac(::Type{Float64}, ::Type{Int64}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_double_64.snls_solve_with_jac_64(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int64},
                                                     n::Int64, m_r::Int64,
                                                     m_c::Int64,
                                                     x::Ptr{Float64},
                                                     y::Ptr{Float64},
                                                     z::Ptr{Float64},
                                                     r::Ptr{Float64},
                                                     g::Ptr{Float64},
                                                     x_stat::Ptr{Int64},
                                                     eval_r::Ptr{Cvoid},
                                                     jr_ne::Int64,
                                                     eval_jr::Ptr{Cvoid},
                                                     w::Ptr{Float64})::Cvoid
end

function snls_solve_with_jac(::Type{Float128}, ::Type{Int32}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_quadruple.snls_solve_with_jac_q(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int32},
                                                    n::Int32, m_r::Int32,
                                                    m_c::Int32,
                                                    x::Ptr{Float128},
                                                    y::Ptr{Float128},
                                                    z::Ptr{Float128},
                                                    r::Ptr{Float128},
                                                    g::Ptr{Float128},
                                                    x_stat::Ptr{Int32},
                                                    eval_r::Ptr{Cvoid},
                                                    jr_ne::Int32,
                                                    eval_jr::Ptr{Cvoid},
                                                    w::Ptr{Float128})::Cvoid
end

function snls_solve_with_jac(::Type{Float128}, ::Type{Int64}, data, userdata,
                             status, n, m_r, m_c, x, y, z, r, g, x_stat, eval_r,
                             jr_ne, eval_jr, w)
  @ccall libgalahad_quadruple_64.snls_solve_with_jac_q_64(data::Ptr{Ptr{Cvoid}},
                                                          userdata::Ptr{Cvoid},
                                                          status::Ptr{Int64},
                                                          n::Int64, m_r::Int64,
                                                          m_c::Int64,
                                                          x::Ptr{Float128},
                                                          y::Ptr{Float128},
                                                          z::Ptr{Float128},
                                                          r::Ptr{Float128},
                                                          g::Ptr{Float128},
                                                          x_stat::Ptr{Int64},
                                                          eval_r::Ptr{Cvoid},
                                                          jr_ne::Int64,
                                                          eval_jr::Ptr{Cvoid},
                                                          w::Ptr{Float128})::Cvoid
end

export snls_solve_with_jacprod

function snls_solve_with_jacprod(::Type{Float32}, ::Type{Int32}, data, userdata,
                                 status, n, m_r, m_c, x, y, z, r, g, x_stat,
                                 eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_single.snls_solve_with_jacprod_s(data::Ptr{Ptr{Cvoid}},
                                                     userdata::Ptr{Cvoid},
                                                     status::Ptr{Int32},
                                                     n::Int32, m_r::Int32,
                                                     m_c::Int32,
                                                     x::Ptr{Float32},
                                                     y::Ptr{Float32},
                                                     z::Ptr{Float32},
                                                     r::Ptr{Float32},
                                                     g::Ptr{Float32},
                                                     x_stat::Ptr{Int32},
                                                     eval_r::Ptr{Cvoid},
                                                     eval_jr_prod::Ptr{Cvoid},
                                                     eval_jr_scol::Ptr{Cvoid},
                                                     eval_jr_sprod::Ptr{Cvoid},
                                                     w::Ptr{Float32})::Cvoid
end

function snls_solve_with_jacprod(::Type{Float32}, ::Type{Int64}, data, userdata,
                                 status, n, m_r, m_c, x, y, z, r, g, x_stat,
                                 eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_single_64.snls_solve_with_jacprod_s_64(data::Ptr{Ptr{Cvoid}},
                                                           userdata::Ptr{Cvoid},
                                                           status::Ptr{Int64},
                                                           n::Int64, m_r::Int64,
                                                           m_c::Int64,
                                                           x::Ptr{Float32},
                                                           y::Ptr{Float32},
                                                           z::Ptr{Float32},
                                                           r::Ptr{Float32},
                                                           g::Ptr{Float32},
                                                           x_stat::Ptr{Int64},
                                                           eval_r::Ptr{Cvoid},
                                                           eval_jr_prod::Ptr{Cvoid},
                                                           eval_jr_scol::Ptr{Cvoid},
                                                           eval_jr_sprod::Ptr{Cvoid},
                                                           w::Ptr{Float32})::Cvoid
end

function snls_solve_with_jacprod(::Type{Float64}, ::Type{Int32}, data, userdata,
                                 status, n, m_r, m_c, x, y, z, r, g, x_stat,
                                 eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_double.snls_solve_with_jacprod(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid},
                                                   status::Ptr{Int32}, n::Int32,
                                                   m_r::Int32, m_c::Int32,
                                                   x::Ptr{Float64},
                                                   y::Ptr{Float64},
                                                   z::Ptr{Float64},
                                                   r::Ptr{Float64},
                                                   g::Ptr{Float64},
                                                   x_stat::Ptr{Int32},
                                                   eval_r::Ptr{Cvoid},
                                                   eval_jr_prod::Ptr{Cvoid},
                                                   eval_jr_scol::Ptr{Cvoid},
                                                   eval_jr_sprod::Ptr{Cvoid},
                                                   w::Ptr{Float64})::Cvoid
end

function snls_solve_with_jacprod(::Type{Float64}, ::Type{Int64}, data, userdata,
                                 status, n, m_r, m_c, x, y, z, r, g, x_stat,
                                 eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_double_64.snls_solve_with_jacprod_64(data::Ptr{Ptr{Cvoid}},
                                                         userdata::Ptr{Cvoid},
                                                         status::Ptr{Int64},
                                                         n::Int64, m_r::Int64,
                                                         m_c::Int64,
                                                         x::Ptr{Float64},
                                                         y::Ptr{Float64},
                                                         z::Ptr{Float64},
                                                         r::Ptr{Float64},
                                                         g::Ptr{Float64},
                                                         x_stat::Ptr{Int64},
                                                         eval_r::Ptr{Cvoid},
                                                         eval_jr_prod::Ptr{Cvoid},
                                                         eval_jr_scol::Ptr{Cvoid},
                                                         eval_jr_sprod::Ptr{Cvoid},
                                                         w::Ptr{Float64})::Cvoid
end

function snls_solve_with_jacprod(::Type{Float128}, ::Type{Int32}, data,
                                 userdata, status, n, m_r, m_c, x, y, z, r, g,
                                 x_stat, eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_quadruple.snls_solve_with_jacprod_q(data::Ptr{Ptr{Cvoid}},
                                                        userdata::Ptr{Cvoid},
                                                        status::Ptr{Int32},
                                                        n::Int32, m_r::Int32,
                                                        m_c::Int32,
                                                        x::Ptr{Float128},
                                                        y::Ptr{Float128},
                                                        z::Ptr{Float128},
                                                        r::Ptr{Float128},
                                                        g::Ptr{Float128},
                                                        x_stat::Ptr{Int32},
                                                        eval_r::Ptr{Cvoid},
                                                        eval_jr_prod::Ptr{Cvoid},
                                                        eval_jr_scol::Ptr{Cvoid},
                                                        eval_jr_sprod::Ptr{Cvoid},
                                                        w::Ptr{Float128})::Cvoid
end

function snls_solve_with_jacprod(::Type{Float128}, ::Type{Int64}, data,
                                 userdata, status, n, m_r, m_c, x, y, z, r, g,
                                 x_stat, eval_r, eval_jr_prod, eval_jr_scol,
                                 eval_jr_sprod, w)
  @ccall libgalahad_quadruple_64.snls_solve_with_jacprod_q_64(data::Ptr{Ptr{Cvoid}},
                                                              userdata::Ptr{Cvoid},
                                                              status::Ptr{Int64},
                                                              n::Int64,
                                                              m_r::Int64,
                                                              m_c::Int64,
                                                              x::Ptr{Float128},
                                                              y::Ptr{Float128},
                                                              z::Ptr{Float128},
                                                              r::Ptr{Float128},
                                                              g::Ptr{Float128},
                                                              x_stat::Ptr{Int64},
                                                              eval_r::Ptr{Cvoid},
                                                              eval_jr_prod::Ptr{Cvoid},
                                                              eval_jr_scol::Ptr{Cvoid},
                                                              eval_jr_sprod::Ptr{Cvoid},
                                                              w::Ptr{Float128})::Cvoid
end

export snls_solve_reverse_with_jac

function snls_solve_reverse_with_jac(::Type{Float32}, ::Type{Int32}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_single.snls_solve_reverse_with_jac_s(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32},
                                                         eval_status::Ptr{Int32},
                                                         n::Int32, m_r::Int32,
                                                         m_c::Int32,
                                                         x::Ptr{Float32},
                                                         y::Ptr{Float32},
                                                         z::Ptr{Float32},
                                                         r::Ptr{Float32},
                                                         g::Ptr{Float32},
                                                         x_stat::Ptr{Int32},
                                                         jr_ne::Int32,
                                                         jr_val::Ptr{Float32},
                                                         w::Ptr{Float32})::Cvoid
end

function snls_solve_reverse_with_jac(::Type{Float32}, ::Type{Int64}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_single_64.snls_solve_reverse_with_jac_s_64(data::Ptr{Ptr{Cvoid}},
                                                               status::Ptr{Int64},
                                                               eval_status::Ptr{Int64},
                                                               n::Int64,
                                                               m_r::Int64,
                                                               m_c::Int64,
                                                               x::Ptr{Float32},
                                                               y::Ptr{Float32},
                                                               z::Ptr{Float32},
                                                               r::Ptr{Float32},
                                                               g::Ptr{Float32},
                                                               x_stat::Ptr{Int64},
                                                               jr_ne::Int64,
                                                               jr_val::Ptr{Float32},
                                                               w::Ptr{Float32})::Cvoid
end

function snls_solve_reverse_with_jac(::Type{Float64}, ::Type{Int32}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_double.snls_solve_reverse_with_jac(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32},
                                                       eval_status::Ptr{Int32},
                                                       n::Int32, m_r::Int32,
                                                       m_c::Int32,
                                                       x::Ptr{Float64},
                                                       y::Ptr{Float64},
                                                       z::Ptr{Float64},
                                                       r::Ptr{Float64},
                                                       g::Ptr{Float64},
                                                       x_stat::Ptr{Int32},
                                                       jr_ne::Int32,
                                                       jr_val::Ptr{Float64},
                                                       w::Ptr{Float64})::Cvoid
end

function snls_solve_reverse_with_jac(::Type{Float64}, ::Type{Int64}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_double_64.snls_solve_reverse_with_jac_64(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int64},
                                                             eval_status::Ptr{Int64},
                                                             n::Int64,
                                                             m_r::Int64,
                                                             m_c::Int64,
                                                             x::Ptr{Float64},
                                                             y::Ptr{Float64},
                                                             z::Ptr{Float64},
                                                             r::Ptr{Float64},
                                                             g::Ptr{Float64},
                                                             x_stat::Ptr{Int64},
                                                             jr_ne::Int64,
                                                             jr_val::Ptr{Float64},
                                                             w::Ptr{Float64})::Cvoid
end

function snls_solve_reverse_with_jac(::Type{Float128}, ::Type{Int32}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_quadruple.snls_solve_reverse_with_jac_q(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int32},
                                                            eval_status::Ptr{Int32},
                                                            n::Int32,
                                                            m_r::Int32,
                                                            m_c::Int32,
                                                            x::Ptr{Float128},
                                                            y::Ptr{Float128},
                                                            z::Ptr{Float128},
                                                            r::Ptr{Float128},
                                                            g::Ptr{Float128},
                                                            x_stat::Ptr{Int32},
                                                            jr_ne::Int32,
                                                            jr_val::Ptr{Float128},
                                                            w::Ptr{Float128})::Cvoid
end

function snls_solve_reverse_with_jac(::Type{Float128}, ::Type{Int64}, data,
                                     status, eval_status, n, m_r, m_c, x, y, z,
                                     r, g, x_stat, jr_ne, jr_val, w)
  @ccall libgalahad_quadruple_64.snls_solve_reverse_with_jac_q_64(data::Ptr{Ptr{Cvoid}},
                                                                  status::Ptr{Int64},
                                                                  eval_status::Ptr{Int64},
                                                                  n::Int64,
                                                                  m_r::Int64,
                                                                  m_c::Int64,
                                                                  x::Ptr{Float128},
                                                                  y::Ptr{Float128},
                                                                  z::Ptr{Float128},
                                                                  r::Ptr{Float128},
                                                                  g::Ptr{Float128},
                                                                  x_stat::Ptr{Int64},
                                                                  jr_ne::Int64,
                                                                  jr_val::Ptr{Float128},
                                                                  w::Ptr{Float128})::Cvoid
end

export snls_solve_reverse_with_jacprod

function snls_solve_reverse_with_jacprod(::Type{Float32}, ::Type{Int32}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_single.snls_solve_reverse_with_jacprod_s(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Int32},
                                                             eval_status::Ptr{Int32},
                                                             n::Int32,
                                                             m_r::Int32,
                                                             m_c::Int32,
                                                             x::Ptr{Float32},
                                                             y::Ptr{Float32},
                                                             z::Ptr{Float32},
                                                             r::Ptr{Float32},
                                                             g::Ptr{Float32},
                                                             x_stat::Ptr{Int32},
                                                             v::Ptr{Float32},
                                                             iv::Ptr{Int32},
                                                             lvl::Ptr{Int32},
                                                             lvu::Ptr{Int32},
                                                             index::Ptr{Int32},
                                                             p::Ptr{Float32},
                                                             ip::Ptr{Int32},
                                                             lp::Int32,
                                                             w::Ptr{Float32})::Cvoid
end

function snls_solve_reverse_with_jacprod(::Type{Float32}, ::Type{Int64}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_single_64.snls_solve_reverse_with_jacprod_s_64(data::Ptr{Ptr{Cvoid}},
                                                                   status::Ptr{Int64},
                                                                   eval_status::Ptr{Int64},
                                                                   n::Int64,
                                                                   m_r::Int64,
                                                                   m_c::Int64,
                                                                   x::Ptr{Float32},
                                                                   y::Ptr{Float32},
                                                                   z::Ptr{Float32},
                                                                   r::Ptr{Float32},
                                                                   g::Ptr{Float32},
                                                                   x_stat::Ptr{Int64},
                                                                   v::Ptr{Float32},
                                                                   iv::Ptr{Int64},
                                                                   lvl::Ptr{Int64},
                                                                   lvu::Ptr{Int64},
                                                                   index::Ptr{Int64},
                                                                   p::Ptr{Float32},
                                                                   ip::Ptr{Int64},
                                                                   lp::Int64,
                                                                   w::Ptr{Float32})::Cvoid
end

function snls_solve_reverse_with_jacprod(::Type{Float64}, ::Type{Int32}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_double.snls_solve_reverse_with_jacprod(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int32},
                                                           eval_status::Ptr{Int32},
                                                           n::Int32, m_r::Int32,
                                                           m_c::Int32,
                                                           x::Ptr{Float64},
                                                           y::Ptr{Float64},
                                                           z::Ptr{Float64},
                                                           r::Ptr{Float64},
                                                           g::Ptr{Float64},
                                                           x_stat::Ptr{Int32},
                                                           v::Ptr{Float64},
                                                           iv::Ptr{Int32},
                                                           lvl::Ptr{Int32},
                                                           lvu::Ptr{Int32},
                                                           index::Ptr{Int32},
                                                           p::Ptr{Float64},
                                                           ip::Ptr{Int32},
                                                           lp::Int32,
                                                           w::Ptr{Float64})::Cvoid
end

function snls_solve_reverse_with_jacprod(::Type{Float64}, ::Type{Int64}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_double_64.snls_solve_reverse_with_jacprod_64(data::Ptr{Ptr{Cvoid}},
                                                                 status::Ptr{Int64},
                                                                 eval_status::Ptr{Int64},
                                                                 n::Int64,
                                                                 m_r::Int64,
                                                                 m_c::Int64,
                                                                 x::Ptr{Float64},
                                                                 y::Ptr{Float64},
                                                                 z::Ptr{Float64},
                                                                 r::Ptr{Float64},
                                                                 g::Ptr{Float64},
                                                                 x_stat::Ptr{Int64},
                                                                 v::Ptr{Float64},
                                                                 iv::Ptr{Int64},
                                                                 lvl::Ptr{Int64},
                                                                 lvu::Ptr{Int64},
                                                                 index::Ptr{Int64},
                                                                 p::Ptr{Float64},
                                                                 ip::Ptr{Int64},
                                                                 lp::Int64,
                                                                 w::Ptr{Float64})::Cvoid
end

function snls_solve_reverse_with_jacprod(::Type{Float128}, ::Type{Int32}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_quadruple.snls_solve_reverse_with_jacprod_q(data::Ptr{Ptr{Cvoid}},
                                                                status::Ptr{Int32},
                                                                eval_status::Ptr{Int32},
                                                                n::Int32,
                                                                m_r::Int32,
                                                                m_c::Int32,
                                                                x::Ptr{Float128},
                                                                y::Ptr{Float128},
                                                                z::Ptr{Float128},
                                                                r::Ptr{Float128},
                                                                g::Ptr{Float128},
                                                                x_stat::Ptr{Int32},
                                                                v::Ptr{Float128},
                                                                iv::Ptr{Int32},
                                                                lvl::Ptr{Int32},
                                                                lvu::Ptr{Int32},
                                                                index::Ptr{Int32},
                                                                p::Ptr{Float128},
                                                                ip::Ptr{Int32},
                                                                lp::Int32,
                                                                w::Ptr{Float128})::Cvoid
end

function snls_solve_reverse_with_jacprod(::Type{Float128}, ::Type{Int64}, data,
                                         status, eval_status, n, m_r, m_c, x, y,
                                         z, r, g, x_stat, v, iv, lvl, lvu,
                                         index, p, ip, lp, w)
  @ccall libgalahad_quadruple_64.snls_solve_reverse_with_jacprod_q_64(data::Ptr{Ptr{Cvoid}},
                                                                      status::Ptr{Int64},
                                                                      eval_status::Ptr{Int64},
                                                                      n::Int64,
                                                                      m_r::Int64,
                                                                      m_c::Int64,
                                                                      x::Ptr{Float128},
                                                                      y::Ptr{Float128},
                                                                      z::Ptr{Float128},
                                                                      r::Ptr{Float128},
                                                                      g::Ptr{Float128},
                                                                      x_stat::Ptr{Int64},
                                                                      v::Ptr{Float128},
                                                                      iv::Ptr{Int64},
                                                                      lvl::Ptr{Int64},
                                                                      lvu::Ptr{Int64},
                                                                      index::Ptr{Int64},
                                                                      p::Ptr{Float128},
                                                                      ip::Ptr{Int64},
                                                                      lp::Int64,
                                                                      w::Ptr{Float128})::Cvoid
end

export snls_information

function snls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.snls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{snls_inform_type{Float32,
                                                                           Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function snls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.snls_information_s_64(data::Ptr{Ptr{Cvoid}},
                                                    inform::Ptr{snls_inform_type{Float32,
                                                                                 Int64}},
                                                    status::Ptr{Int64})::Cvoid
end

function snls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.snls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{snls_inform_type{Float64,
                                                                         Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function snls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.snls_information_64(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{snls_inform_type{Float64,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function snls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.snls_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{snls_inform_type{Float128,
                                                                              Int32}},
                                                 status::Ptr{Int32})::Cvoid
end

function snls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.snls_information_q_64(data::Ptr{Ptr{Cvoid}},
                                                       inform::Ptr{snls_inform_type{Float128,
                                                                                    Int64}},
                                                       status::Ptr{Int64})::Cvoid
end

export snls_terminate

function snls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.snls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{snls_control_type{Float32,
                                                                           Int32}},
                                            inform::Ptr{snls_inform_type{Float32,
                                                                         Int32}})::Cvoid
end

function snls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.snls_terminate_s_64(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{snls_control_type{Float32,
                                                                                 Int64}},
                                                  inform::Ptr{snls_inform_type{Float32,
                                                                               Int64}})::Cvoid
end

function snls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.snls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{snls_control_type{Float64,
                                                                         Int32}},
                                          inform::Ptr{snls_inform_type{Float64,
                                                                       Int32}})::Cvoid
end

function snls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.snls_terminate_64(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{snls_control_type{Float64,
                                                                               Int64}},
                                                inform::Ptr{snls_inform_type{Float64,
                                                                             Int64}})::Cvoid
end

function snls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.snls_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{snls_control_type{Float128,
                                                                              Int32}},
                                               inform::Ptr{snls_inform_type{Float128,
                                                                            Int32}})::Cvoid
end

function snls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.snls_terminate_q_64(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{snls_control_type{Float128,
                                                                                    Int64}},
                                                     inform::Ptr{snls_inform_type{Float128,
                                                                                  Int64}})::Cvoid
end

function run_sif(::Val{:snls}, ::Val{:single}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsnls_sif_single()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end

function run_sif(::Val{:snls}, ::Val{:double}, path_libsif::String,
                 path_outsdif::String)
  cmd = setup_env_lbt(`$(GALAHAD_jll.runsnls_sif_double()) $path_libsif $path_outsdif`)
  run(cmd)
  return nothing
end
