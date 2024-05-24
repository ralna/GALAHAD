export sha_control_type

struct sha_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  approximation_algorithm::Cint
  dense_linear_solver::Cint
  sparse_row::Cint
  extra_differences::Cint
  recursion_max::Cint
  recursion_entries_required::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export sha_inform_type

struct sha_inform_type
  status::Cint
  alloc_status::Cint
  max_degree::Cint
  differences_needed::Cint
  max_reduced_degree::Cint
  approximation_algorithm_used::Cint
  bad_row::Cint
  bad_alloc::NTuple{81,Cchar}
end

export sha_initialize_s

function sha_initialize_s(data, control, status)
  @ccall libgalahad_single.sha_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sha_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export sha_initialize

function sha_initialize(data, control, status)
  @ccall libgalahad_double.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sha_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export sha_reset_control_s

function sha_reset_control_s(control, data, status)
  @ccall libgalahad_single.sha_reset_control_s(control::Ptr{sha_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export sha_reset_control

function sha_reset_control(control, data, status)
  @ccall libgalahad_double.sha_reset_control(control::Ptr{sha_control_type},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export sha_analyse_matrix_s

function sha_analyse_matrix_s(control, data, status, n, ne, row, col, m)
  @ccall libgalahad_single.sha_analyse_matrix_s(control::Ptr{sha_control_type},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, ne::Cint, row::Ptr{Cint},
                                                col::Ptr{Cint}, m::Ptr{Cint})::Cvoid
end

export sha_analyse_matrix

function sha_analyse_matrix(control, data, status, n, ne, row, col, m)
  @ccall libgalahad_double.sha_analyse_matrix(control::Ptr{sha_control_type},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, ne::Cint, row::Ptr{Cint},
                                              col::Ptr{Cint}, m::Ptr{Cint})::Cvoid
end

export sha_recover_matrix_s

function sha_recover_matrix_s(data, status, ne, m, ls1, ls2, strans, ly1, ly2, ytrans, val,
                              precedence)
  @ccall libgalahad_single.sha_recover_matrix_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                ne::Cint, m::Cint, ls1::Cint, ls2::Cint,
                                                strans::Ptr{Ptr{Float32}}, ly1::Cint,
                                                ly2::Cint, ytrans::Ptr{Ptr{Float32}},
                                                val::Ptr{Float32},
                                                precedence::Ptr{Cint})::Cvoid
end

export sha_recover_matrix

function sha_recover_matrix(data, status, ne, m, ls1, ls2, strans, ly1, ly2, ytrans, val,
                            precedence)
  @ccall libgalahad_double.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              ne::Cint, m::Cint, ls1::Cint, ls2::Cint,
                                              strans::Ptr{Ptr{Float64}}, ly1::Cint,
                                              ly2::Cint, ytrans::Ptr{Ptr{Float64}},
                                              val::Ptr{Float64},
                                              precedence::Ptr{Cint})::Cvoid
end

export sha_information_s

function sha_information_s(data, inform, status)
  @ccall libgalahad_single.sha_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{sha_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export sha_information

function sha_information(data, inform, status)
  @ccall libgalahad_double.sha_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sha_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export sha_terminate_s

function sha_terminate_s(data, control, inform)
  @ccall libgalahad_single.sha_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sha_control_type},
                                           inform::Ptr{sha_inform_type})::Cvoid
end

export sha_terminate

function sha_terminate(data, control, inform)
  @ccall libgalahad_double.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sha_control_type},
                                         inform::Ptr{sha_inform_type})::Cvoid
end
