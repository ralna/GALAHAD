export psls_control_type

mutable struct psls_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  preconditioner::Cint
  semi_bandwidth::Cint
  scaling::Cint
  ordering::Cint
  max_col::Cint
  icfs_vectors::Cint
  mi28_lsize::Cint
  mi28_rsize::Cint
  min_diagonal::T
  new_structure::Bool
  get_semi_bandwidth::Bool
  get_norm_residual::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  definite_linear_solver::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sls_control::sls_control_type{T}
  mi28_control::mi28_control{T}

  psls_control_type{T}() where T = new()
end

export psls_time_type

mutable struct psls_time_type{T}
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

  psls_time_type{T}() where T = new()
end

export psls_inform_type

mutable struct psls_inform_type{T}
  status::Cint
  alloc_status::Cint
  analyse_status::Cint
  factorize_status::Cint
  solve_status::Cint
  factorization_integer::Int64
  factorization_real::Int64
  preconditioner::Cint
  semi_bandwidth::Cint
  reordered_semi_bandwidth::Cint
  out_of_range::Cint
  duplicates::Cint
  upper::Cint
  missing_diagonals::Cint
  semi_bandwidth_used::Cint
  neg1::Cint
  neg2::Cint
  perturbed::Bool
  fill_in_ratio::T
  norm_residual::T
  bad_alloc::NTuple{81,Cchar}
  mc61_info::NTuple{10,Cint}
  mc61_rinfo::NTuple{15,T}
  time::psls_time_type{T}
  sls_inform::sls_inform_type{T}
  mi28_info::mi28_info{T}

  psls_inform_type{T}() where T = new()
end

export psls_initialize_s

function psls_initialize_s(data, control, status)
  @ccall libgalahad_single.psls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{psls_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export psls_initialize

function psls_initialize(data, control, status)
  @ccall libgalahad_double.psls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{psls_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export psls_read_specfile_s

function psls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.psls_read_specfile_s(control::Ref{psls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export psls_read_specfile

function psls_read_specfile(control, specfile)
  @ccall libgalahad_double.psls_read_specfile(control::Ref{psls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export psls_import_s

function psls_import_s(control, data, status, n, type, ne, row, col, ptr)
  @ccall libgalahad_single.psls_import_s(control::Ref{psls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         type::Ptr{Cchar}, ne::Cint, row::Ptr{Cint},
                                         col::Ptr{Cint}, ptr::Ptr{Cint})::Cvoid
end

export psls_import

function psls_import(control, data, status, n, type, ne, row, col, ptr)
  @ccall libgalahad_double.psls_import(control::Ref{psls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       type::Ptr{Cchar}, ne::Cint, row::Ptr{Cint},
                                       col::Ptr{Cint}, ptr::Ptr{Cint})::Cvoid
end

export psls_reset_control_s

function psls_reset_control_s(control, data, status)
  @ccall libgalahad_single.psls_reset_control_s(control::Ref{psls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export psls_reset_control

function psls_reset_control(control, data, status)
  @ccall libgalahad_double.psls_reset_control(control::Ref{psls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export psls_form_preconditioner_s

function psls_form_preconditioner_s(data, status, ne, val)
  @ccall libgalahad_single.psls_form_preconditioner_s(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, ne::Cint,
                                                      val::Ptr{Float32})::Cvoid
end

export psls_form_preconditioner

function psls_form_preconditioner(data, status, ne, val)
  @ccall libgalahad_double.psls_form_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, ne::Cint,
                                                    val::Ptr{Float64})::Cvoid
end

export psls_form_subset_preconditioner_s

function psls_form_subset_preconditioner_s(data, status, ne, val, n_sub, sub)
  @ccall libgalahad_single.psls_form_subset_preconditioner_s(data::Ptr{Ptr{Cvoid}},
                                                             status::Ptr{Cint}, ne::Cint,
                                                             val::Ptr{Float32},
                                                             n_sub::Cint,
                                                             sub::Ptr{Cint})::Cvoid
end

export psls_form_subset_preconditioner

function psls_form_subset_preconditioner(data, status, ne, val, n_sub, sub)
  @ccall libgalahad_double.psls_form_subset_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Cint}, ne::Cint,
                                                           val::Ptr{Float64},
                                                           n_sub::Cint,
                                                           sub::Ptr{Cint})::Cvoid
end

export psls_update_preconditioner_s

function psls_update_preconditioner_s(data, status, ne, val, n_del, del)
  @ccall libgalahad_single.psls_update_preconditioner_s(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint}, ne::Cint,
                                                        val::Ptr{Float32}, n_del::Cint,
                                                        del::Ptr{Cint})::Cvoid
end

export psls_update_preconditioner

function psls_update_preconditioner(data, status, ne, val, n_del, del)
  @ccall libgalahad_double.psls_update_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, ne::Cint,
                                                      val::Ptr{Float64}, n_del::Cint,
                                                      del::Ptr{Cint})::Cvoid
end

export psls_apply_preconditioner_s

function psls_apply_preconditioner_s(data, status, n, sol)
  @ccall libgalahad_single.psls_apply_preconditioner_s(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint}, n::Cint,
                                                       sol::Ptr{Float32})::Cvoid
end

export psls_apply_preconditioner

function psls_apply_preconditioner(data, status, n, sol)
  @ccall libgalahad_double.psls_apply_preconditioner(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Cint}, n::Cint,
                                                     sol::Ptr{Float64})::Cvoid
end

export psls_information_s

function psls_information_s(data, inform, status)
  @ccall libgalahad_single.psls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{psls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export psls_information

function psls_information(data, inform, status)
  @ccall libgalahad_double.psls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{psls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export psls_terminate_s

function psls_terminate_s(data, control, inform)
  @ccall libgalahad_single.psls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{psls_control_type{Float32}},
                                            inform::Ref{psls_inform_type{Float32}})::Cvoid
end

export psls_terminate

function psls_terminate(data, control, inform)
  @ccall libgalahad_double.psls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{psls_control_type{Float64}},
                                          inform::Ref{psls_inform_type{Float64}})::Cvoid
end
