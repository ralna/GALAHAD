export psls_control_type

struct psls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  preconditioner::INT
  semi_bandwidth::INT
  scaling::INT
  ordering::INT
  max_col::INT
  icfs_vectors::INT
  mi28_lsize::INT
  mi28_rsize::INT
  min_diagonal::T
  new_structure::Bool
  get_semi_bandwidth::Bool
  get_norm_residual::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T,INT}
  mi28_control::mi28_control{T,INT}
end

export psls_time_type

struct psls_time_type{T}
  total::Float32
  assemble::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
  update::Float32
  clock_total::T
  clock_assemble::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
  clock_update::T
end

export psls_inform_type

struct psls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  analyse_status::INT
  factorize_status::INT
  solve_status::INT
  factorization_integer::Int64
  factorization_real::Int64
  preconditioner::INT
  semi_bandwidth::INT
  reordered_semi_bandwidth::INT
  out_of_range::INT
  duplicates::INT
  upper::INT
  missing_diagonals::INT
  semi_bandwidth_used::INT
  neg1::INT
  neg2::INT
  perturbed::Bool
  fill_in_ratio::T
  norm_residual::T
  bad_alloc::NTuple{81,Cchar}
  mc61_info::NTuple{10,INT}
  mc61_rinfo::NTuple{15,T}
  time::psls_time_type{T}
  sls_inform::sls_inform_type{T,INT}
  mi28_info::mi28_info{T,INT}
end

export psls_initialize

function psls_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{psls_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function psls_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{psls_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function psls_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{psls_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function psls_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{psls_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function psls_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{psls_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function psls_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{psls_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export psls_read_specfile

function psls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.psls_read_specfile(control::Ptr{psls_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function psls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.psls_read_specfile(control::Ptr{psls_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function psls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.psls_read_specfile(control::Ptr{psls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function psls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.psls_read_specfile(control::Ptr{psls_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function psls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.psls_read_specfile(control::Ptr{psls_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function psls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.psls_read_specfile(control::Ptr{psls_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export psls_import

function psls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_single.psls_import(control::Ptr{psls_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       type::Ptr{Cchar}, ne::Int32, row::Ptr{Int32},
                                       col::Ptr{Int32}, ptr::Ptr{Int32})::Cvoid
end

function psls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_single_64.psls_import(control::Ptr{psls_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, type::Ptr{Cchar}, ne::Int64,
                                          row::Ptr{Int64}, col::Ptr{Int64},
                                          ptr::Ptr{Int64})::Cvoid
end

function psls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_double.psls_import(control::Ptr{psls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       type::Ptr{Cchar}, ne::Int32, row::Ptr{Int32},
                                       col::Ptr{Int32}, ptr::Ptr{Int32})::Cvoid
end

function psls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_double_64.psls_import(control::Ptr{psls_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, type::Ptr{Cchar}, ne::Int64,
                                          row::Ptr{Int64}, col::Ptr{Int64},
                                          ptr::Ptr{Int64})::Cvoid
end

function psls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_quadruple.psls_import(control::Ptr{psls_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, type::Ptr{Cchar}, ne::Int32,
                                          row::Ptr{Int32}, col::Ptr{Int32},
                                          ptr::Ptr{Int32})::Cvoid
end

function psls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, type, ne,
                     row, col, ptr)
  @ccall libgalahad_quadruple_64.psls_import(control::Ptr{psls_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, type::Ptr{Cchar}, ne::Int64,
                                             row::Ptr{Int64}, col::Ptr{Int64},
                                             ptr::Ptr{Int64})::Cvoid
end

export psls_reset_control

function psls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.psls_reset_control(control::Ptr{psls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function psls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.psls_reset_control(control::Ptr{psls_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function psls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.psls_reset_control(control::Ptr{psls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function psls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.psls_reset_control(control::Ptr{psls_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function psls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.psls_reset_control(control::Ptr{psls_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function psls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.psls_reset_control(control::Ptr{psls_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export psls_form_preconditioner

function psls_form_preconditioner(::Type{Float32}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_single.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, ne::Int32,
                                                    val::Ptr{Float32})::Cvoid
end

function psls_form_preconditioner(::Type{Float32}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_single_64.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, ne::Int64,
                                                       val::Ptr{Float32})::Cvoid
end

function psls_form_preconditioner(::Type{Float64}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_double.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, ne::Int32,
                                                    val::Ptr{Float64})::Cvoid
end

function psls_form_preconditioner(::Type{Float64}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_double_64.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, ne::Int64,
                                                       val::Ptr{Float64})::Cvoid
end

function psls_form_preconditioner(::Type{Float128}, ::Type{Int32}, data, status, ne, val)
  @ccall libgalahad_quadruple.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32}, ne::Int32,
                                                       val::Ptr{Float128})::Cvoid
end

function psls_form_preconditioner(::Type{Float128}, ::Type{Int64}, data, status, ne, val)
  @ccall libgalahad_quadruple_64.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64}, ne::Int64,
                                                          val::Ptr{Float128})::Cvoid
end

export psls_form_subset_preconditioner

function psls_form_subset_preconditioner(::Type{Float32}, ::Type{Int32}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_single.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int32}, ne::Int32,
                                                           val::Ptr{Float32}, n_sub::Int32,
                                                           sub::Ptr{Int32})::Cvoid
end

function psls_form_subset_preconditioner(::Type{Float32}, ::Type{Int64}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_single_64.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int64}, ne::Int64,
                                                              val::Ptr{Float32},
                                                              n_sub::Int64,
                                                              sub::Ptr{Int64})::Cvoid
end

function psls_form_subset_preconditioner(::Type{Float64}, ::Type{Int32}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_double.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int32}, ne::Int32,
                                                           val::Ptr{Float64}, n_sub::Int32,
                                                           sub::Ptr{Int32})::Cvoid
end

function psls_form_subset_preconditioner(::Type{Float64}, ::Type{Int64}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_double_64.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int64}, ne::Int64,
                                                              val::Ptr{Float64},
                                                              n_sub::Int64,
                                                              sub::Ptr{Int64})::Cvoid
end

function psls_form_subset_preconditioner(::Type{Float128}, ::Type{Int32}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_quadruple.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                              status::Ptr{Int32}, ne::Int32,
                                                              val::Ptr{Float128},
                                                              n_sub::Int32,
                                                              sub::Ptr{Int32})::Cvoid
end

function psls_form_subset_preconditioner(::Type{Float128}, ::Type{Int64}, data, status, ne,
                                         val, n_sub, sub)
  @ccall libgalahad_quadruple_64.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                                 status::Ptr{Int64},
                                                                 ne::Int64,
                                                                 val::Ptr{Float128},
                                                                 n_sub::Int64,
                                                                 sub::Ptr{Int64})::Cvoid
end

export psls_update_preconditioner

function psls_update_preconditioner(::Type{Float32}, ::Type{Int32}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_single.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, ne::Int32,
                                                      val::Ptr{Float32}, n_del::Int32,
                                                      del::Ptr{Int32})::Cvoid
end

function psls_update_preconditioner(::Type{Float32}, ::Type{Int64}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_single_64.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, ne::Int64,
                                                         val::Ptr{Float32}, n_del::Int64,
                                                         del::Ptr{Int64})::Cvoid
end

function psls_update_preconditioner(::Type{Float64}, ::Type{Int32}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_double.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, ne::Int32,
                                                      val::Ptr{Float64}, n_del::Int32,
                                                      del::Ptr{Int32})::Cvoid
end

function psls_update_preconditioner(::Type{Float64}, ::Type{Int64}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_double_64.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, ne::Int64,
                                                         val::Ptr{Float64}, n_del::Int64,
                                                         del::Ptr{Int64})::Cvoid
end

function psls_update_preconditioner(::Type{Float128}, ::Type{Int32}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_quadruple.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32}, ne::Int32,
                                                         val::Ptr{Float128}, n_del::Int32,
                                                         del::Ptr{Int32})::Cvoid
end

function psls_update_preconditioner(::Type{Float128}, ::Type{Int64}, data, status, ne, val,
                                    n_del, del)
  @ccall libgalahad_quadruple_64.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64}, ne::Int64,
                                                            val::Ptr{Float128},
                                                            n_del::Int64,
                                                            del::Ptr{Int64})::Cvoid
end

export psls_apply_preconditioner

function psls_apply_preconditioner(::Type{Float32}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_single.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n::Int32,
                                                     sol::Ptr{Float32})::Cvoid
end

function psls_apply_preconditioner(::Type{Float32}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_single_64.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n::Int64,
                                                        sol::Ptr{Float32})::Cvoid
end

function psls_apply_preconditioner(::Type{Float64}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_double.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n::Int32,
                                                     sol::Ptr{Float64})::Cvoid
end

function psls_apply_preconditioner(::Type{Float64}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_double_64.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n::Int64,
                                                        sol::Ptr{Float64})::Cvoid
end

function psls_apply_preconditioner(::Type{Float128}, ::Type{Int32}, data, status, n, sol)
  @ccall libgalahad_quadruple.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32}, n::Int32,
                                                        sol::Ptr{Float128})::Cvoid
end

function psls_apply_preconditioner(::Type{Float128}, ::Type{Int64}, data, status, n, sol)
  @ccall libgalahad_quadruple_64.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int64}, n::Int64,
                                                           sol::Ptr{Float128})::Cvoid
end

export psls_information

function psls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.psls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{psls_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function psls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.psls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{psls_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function psls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.psls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{psls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function psls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.psls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{psls_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function psls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.psls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{psls_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function psls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.psls_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{psls_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export psls_terminate

function psls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{psls_control_type{Float32,Int32}},
                                          inform::Ptr{psls_inform_type{Float32,Int32}})::Cvoid
end

function psls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{psls_control_type{Float32,Int64}},
                                             inform::Ptr{psls_inform_type{Float32,Int64}})::Cvoid
end

function psls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{psls_control_type{Float64,Int32}},
                                          inform::Ptr{psls_inform_type{Float64,Int32}})::Cvoid
end

function psls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{psls_control_type{Float64,Int64}},
                                             inform::Ptr{psls_inform_type{Float64,Int64}})::Cvoid
end

function psls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{psls_control_type{Float128,Int32}},
                                             inform::Ptr{psls_inform_type{Float128,Int32}})::Cvoid
end

function psls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{psls_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{psls_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
