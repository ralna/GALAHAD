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
  average_off_diagonals::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export sha_inform_type

struct sha_inform_type{T}
  status::Cint
  alloc_status::Cint
  max_degree::Cint
  differences_needed::Cint
  max_reduced_degree::Cint
  approximation_algorithm_used::Cint
  bad_row::Cint
  max_off_diagonal_difference::T
  bad_alloc::NTuple{81,Cchar}
end

export sha_initialize

function sha_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.sha_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sha_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function sha_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sha_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export sha_reset_control

function sha_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.sha_reset_control_s(control::Ptr{sha_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function sha_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.sha_reset_control(control::Ptr{sha_control_type},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export sha_analyse_matrix

function sha_analyse_matrix(::Type{Float32}, control, data, status, n, ne, row, col, m)
  @ccall libgalahad_single.sha_analyse_matrix_s(control::Ptr{sha_control_type},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, ne::Cint, row::Ptr{Cint},
                                                col::Ptr{Cint}, m::Ptr{Cint})::Cvoid
end

function sha_analyse_matrix(::Type{Float64}, control, data, status, n, ne, row, col, m)
  @ccall libgalahad_double.sha_analyse_matrix(control::Ptr{sha_control_type},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, ne::Cint, row::Ptr{Cint},
                                              col::Ptr{Cint}, m::Ptr{Cint})::Cvoid
end

export sha_recover_matrix

function sha_recover_matrix(::Type{Float32}, data, status, ne, m, ls1, ls2, strans, ly1,
                            ly2, ytrans, val, precedence)
  @ccall libgalahad_single.sha_recover_matrix_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                ne::Cint, m::Cint, ls1::Cint, ls2::Cint,
                                                strans::Ptr{Ptr{Float32}}, ly1::Cint,
                                                ly2::Cint, ytrans::Ptr{Ptr{Float32}},
                                                val::Ptr{Float32},
                                                precedence::Ptr{Cint})::Cvoid
end

function sha_recover_matrix(::Type{Float64}, data, status, ne, m, ls1, ls2, strans, ly1,
                            ly2, ytrans, val, precedence)
  @ccall libgalahad_double.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              ne::Cint, m::Cint, ls1::Cint, ls2::Cint,
                                              strans::Ptr{Ptr{Float64}}, ly1::Cint,
                                              ly2::Cint, ytrans::Ptr{Ptr{Float64}},
                                              val::Ptr{Float64},
                                              precedence::Ptr{Cint})::Cvoid
end

export sha_information

function sha_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.sha_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{sha_inform_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function sha_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.sha_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sha_inform_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export sha_terminate

function sha_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.sha_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sha_control_type},
                                           inform::Ptr{sha_inform_type{Float32}})::Cvoid
end

function sha_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sha_control_type},
                                         inform::Ptr{sha_inform_type{Float64}})::Cvoid
end
