export sha_control_type

struct sha_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  approximation_algorithm::INT
  dense_linear_solver::INT
  extra_differences::INT
  sparse_row::INT
  recursion_max::INT
  recursion_entries_required::INT
  average_off_diagonals::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export sha_inform_type

struct sha_inform_type{T,INT}
  status::INT
  alloc_status::INT
  max_degree::INT
  differences_needed::INT
  max_reduced_degree::INT
  approximation_algorithm_used::INT
  bad_row::INT
  max_off_diagonal_difference::T
  bad_alloc::NTuple{81,Cchar}
end

export sha_initialize

function sha_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sha_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sha_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sha_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sha_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{sha_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function sha_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sha_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function sha_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{sha_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function sha_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{sha_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export sha_reset_control

function sha_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.sha_reset_control(control::Ptr{sha_control_type{Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function sha_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.sha_reset_control(control::Ptr{sha_control_type{Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function sha_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.sha_reset_control(control::Ptr{sha_control_type{Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function sha_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.sha_reset_control(control::Ptr{sha_control_type{Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function sha_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.sha_reset_control(control::Ptr{sha_control_type{Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function sha_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.sha_reset_control(control::Ptr{sha_control_type{Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export sha_analyse_matrix

function sha_analyse_matrix(::Type{Float32}, ::Type{Int32}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_single.sha_analyse_matrix(control::Ptr{sha_control_type{Int32}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, ne::Int32, row::Ptr{Int32},
                                              col::Ptr{Int32}, m::Ptr{Int32})::Cvoid
end

function sha_analyse_matrix(::Type{Float32}, ::Type{Int64}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_single_64.sha_analyse_matrix(control::Ptr{sha_control_type{Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, ne::Int64, row::Ptr{Int64},
                                                 col::Ptr{Int64}, m::Ptr{Int64})::Cvoid
end

function sha_analyse_matrix(::Type{Float64}, ::Type{Int32}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_double.sha_analyse_matrix(control::Ptr{sha_control_type{Int32}},
                                              data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              n::Int32, ne::Int32, row::Ptr{Int32},
                                              col::Ptr{Int32}, m::Ptr{Int32})::Cvoid
end

function sha_analyse_matrix(::Type{Float64}, ::Type{Int64}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_double_64.sha_analyse_matrix(control::Ptr{sha_control_type{Int64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 n::Int64, ne::Int64, row::Ptr{Int64},
                                                 col::Ptr{Int64}, m::Ptr{Int64})::Cvoid
end

function sha_analyse_matrix(::Type{Float128}, ::Type{Int32}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_quadruple.sha_analyse_matrix(control::Ptr{sha_control_type{Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, ne::Int32, row::Ptr{Int32},
                                                 col::Ptr{Int32}, m::Ptr{Int32})::Cvoid
end

function sha_analyse_matrix(::Type{Float128}, ::Type{Int64}, control, data, status, n, ne,
                            row, col, m)
  @ccall libgalahad_quadruple_64.sha_analyse_matrix(control::Ptr{sha_control_type{Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64, ne::Int64,
                                                    row::Ptr{Int64}, col::Ptr{Int64},
                                                    m::Ptr{Int64})::Cvoid
end

export sha_recover_matrix

function sha_recover_matrix(::Type{Float32}, ::Type{Int32}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_single.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              ne::Int32, m::Int32, ls1::Int32, ls2::Int32,
                                              strans::Ptr{Ptr{Float32}}, ly1::Int32,
                                              ly2::Int32, ytrans::Ptr{Ptr{Float32}},
                                              val::Ptr{Float32},
                                              precedence::Ptr{Int32})::Cvoid
end

function sha_recover_matrix(::Type{Float32}, ::Type{Int64}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_single_64.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 ne::Int64, m::Int64, ls1::Int64,
                                                 ls2::Int64, strans::Ptr{Ptr{Float32}},
                                                 ly1::Int64, ly2::Int64,
                                                 ytrans::Ptr{Ptr{Float32}},
                                                 val::Ptr{Float32},
                                                 precedence::Ptr{Int64})::Cvoid
end

function sha_recover_matrix(::Type{Float64}, ::Type{Int32}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_double.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                              ne::Int32, m::Int32, ls1::Int32, ls2::Int32,
                                              strans::Ptr{Ptr{Float64}}, ly1::Int32,
                                              ly2::Int32, ytrans::Ptr{Ptr{Float64}},
                                              val::Ptr{Float64},
                                              precedence::Ptr{Int32})::Cvoid
end

function sha_recover_matrix(::Type{Float64}, ::Type{Int64}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_double_64.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                 ne::Int64, m::Int64, ls1::Int64,
                                                 ls2::Int64, strans::Ptr{Ptr{Float64}},
                                                 ly1::Int64, ly2::Int64,
                                                 ytrans::Ptr{Ptr{Float64}},
                                                 val::Ptr{Float64},
                                                 precedence::Ptr{Int64})::Cvoid
end

function sha_recover_matrix(::Type{Float128}, ::Type{Int32}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_quadruple.sha_recover_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 ne::Int32, m::Int32, ls1::Int32,
                                                 ls2::Int32, strans::Ptr{Ptr{Float128}},
                                                 ly1::Int32, ly2::Int32,
                                                 ytrans::Ptr{Ptr{Float128}},
                                                 val::Ptr{Float128},
                                                 precedence::Ptr{Int32})::Cvoid
end

function sha_recover_matrix(::Type{Float128}, ::Type{Int64}, data, status, ne, m, ls1, ls2,
                            strans, ly1, ly2, ytrans, val, precedence)
  @ccall libgalahad_quadruple_64.sha_recover_matrix(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, ne::Int64, m::Int64,
                                                    ls1::Int64, ls2::Int64,
                                                    strans::Ptr{Ptr{Float128}}, ly1::Int64,
                                                    ly2::Int64, ytrans::Ptr{Ptr{Float128}},
                                                    val::Ptr{Float128},
                                                    precedence::Ptr{Int64})::Cvoid
end

export sha_information

function sha_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.sha_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sha_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sha_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.sha_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sha_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sha_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.sha_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{sha_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function sha_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.sha_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sha_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function sha_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.sha_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{sha_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function sha_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.sha_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{sha_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export sha_terminate

function sha_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sha_control_type{Int32}},
                                         inform::Ptr{sha_inform_type{Float32,Int32}})::Cvoid
end

function sha_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sha_control_type{Int64}},
                                            inform::Ptr{sha_inform_type{Float32,Int64}})::Cvoid
end

function sha_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{sha_control_type{Int32}},
                                         inform::Ptr{sha_inform_type{Float64,Int32}})::Cvoid
end

function sha_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sha_control_type{Int64}},
                                            inform::Ptr{sha_inform_type{Float64,Int64}})::Cvoid
end

function sha_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{sha_control_type{Int32}},
                                            inform::Ptr{sha_inform_type{Float128,Int32}})::Cvoid
end

function sha_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{sha_control_type{Int64}},
                                               inform::Ptr{sha_inform_type{Float128,Int64}})::Cvoid
end
