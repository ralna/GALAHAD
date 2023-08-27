export sha_control_type

mutable struct sha_control_type
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  approximation_algorithm::Cint
  dense_linear_solver::Cint
  max_sparse_degree::Cint
  extra_differences::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}

  sha_control_type() = new()
end

export sha_inform_type

mutable struct sha_inform_type
  status::Cint
  alloc_status::Cint
  max_degree::Cint
  approximation_algorithm_used::Cint
  differences_needed::Cint
  max_reduced_degree::Cint
  bad_row::Cint
  bad_alloc::NTuple{81,Cchar}

  sha_inform_type() = new()
end

export sha_initialize_s

function sha_initialize_s(data, control, status)
  @ccall libgalahad_single.sha_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{sha_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export sha_initialize

export sha_read_specfile_s

function sha_read_specfile_s(control, specfile)
  @ccall libgalahad_single.sha_read_specfile_s(control::Ref{sha_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export sha_read_specfile

function sha_read_specfile(control, specfile)
  @ccall libgalahad_double.sha_read_specfile(control::Ref{sha_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end


export sha_analyse_matrix_s

function sha_analyse_matrix_s(control, data, status, n, ne, row, col, m)
  @ccall libgalahad_single.sha_analyse_matrix_s(control::Ref{sha_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, ne::Cint,
                                              row::Ptr{Cint}, col::Ptr{Cint},
                                              m::Cint)::Cvoid
end

export sha_analyse_matrix

function sha_analyse_matrix(control, data, status, n, ne, row, col, m)
  @ccall libgalahad_double.sha_analyse_matrix(control::Ref{sha_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, ne::Cint,
                                              row::Ptr{Cint}, col::Ptr{Cint},
                                              m::Cint)::Cvoid
end

export sha_recover_matrix_s

function sha_recover_matrix_s(data, status, ne, m, ls1, ls2, strans, ly1, ly2, ytrans, val, order)
  @ccall libgalahad_single.sha_recover_matrix_s(data::Ptr{Ptr{Cvoid}}, 
                                              status::Ptr{Cint},
                                              ne::Cint, m::Cint, 
                                              ls1::Cint, ls2::Cint, 
                                              strans::Ptr{Float32}, 
                                              ly1::Cint, ly2::Cint, 
                                              ytrans::Ptr{Float32}, 
                                              val::Ptr{Float32}, 
                                              order::Ptr{Cint})::Cvoid
end

export sha_recover_matrix

function sha_recover_matrix(data, status, ne, m, ls1, ls2, strans, ly1, ly2, ytrans, val, order)
  @ccall libgalahad_double.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, 
                                              status::Ptr{Cint},
                                              ne::Cint, m::Cint, 
                                              ls1::Cint, ls2::Cint, 
                                              strans::Ptr{Float64}, 
                                              ly1::Cint, ly2::Cint, 
                                              ytrans::Ptr{Float64}, 
                                              val::Ptr{Float64}, 
                                              order::Ptr{Cint})::Cvoid
end

function sha_initialize(data, control, status)
  @ccall libgalahad_double.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{sha_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export sha_information_s

function sha_information_s(data, inform, status)
  @ccall libgalahad_single.sha_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{sha_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export sha_information

function sha_information(data, inform, status)
  @ccall libgalahad_double.sha_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{sha_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export sha_terminate_s

function sha_terminate_s(data, control, inform)
  @ccall libgalahad_single.sha_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sha_control_type},
                                           inform::Ref{sha_inform_type})::Cvoid
end

export sha_terminate

function sha_terminate(data, control, inform)
  @ccall libgalahad_double.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{sha_control_type},
                                         inform::Ref{sha_inform_type})::Cvoid
end
