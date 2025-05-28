export bsc_control_type

struct bsc_control_type{INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  max_col::INT
  new_a::INT
  extra_space_s::INT
  s_also_by_column::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export bsc_inform_type

struct bsc_inform_type{T,INT}
  status::INT
  alloc_status::INT
  bad_alloc::NTuple{81,Cchar}
  max_col_a::INT
  exceeds_max_col::INT
  time::T
  clock_time::T
end

export bsc_initialize

function bsc_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bsc_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bsc_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bsc_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bsc_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bsc_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bsc_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bsc_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

function bsc_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bsc_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
  new_control = @set control[].f_indexing = true
  control[] = new_control[]
  return Cvoid
end

export bsc_information

function bsc_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.bsc_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bsc_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bsc_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.bsc_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bsc_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bsc_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.bsc_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bsc_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function bsc_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.bsc_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{bsc_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export bsc_terminate

function bsc_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bsc_control_type{Int32}},
                                         inform::Ptr{bsc_inform_type{Float32,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int64}},
                                            inform::Ptr{bsc_inform_type{Float32,Int64}})::Cvoid
end

function bsc_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bsc_control_type{Int32}},
                                         inform::Ptr{bsc_inform_type{Float64,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int64}},
                                            inform::Ptr{bsc_inform_type{Float64,Int64}})::Cvoid
end

function bsc_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bsc_control_type{Int32}},
                                            inform::Ptr{bsc_inform_type{Float128,Int32}})::Cvoid
end

function bsc_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{bsc_control_type{Int64}},
                                               inform::Ptr{bsc_inform_type{Float128,Int64}})::Cvoid
end




export bsc_import

function bsc_import(::Type{Float32}, ::Type{Int32}, control, data, status, 
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_single.bsc_import(control::Ptr{bsc_control_type{Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                      m::Int32, n::Int32, A_type::Ptr{Cchar}, 
                                      A_ne::Int32, A_row::Ptr{Int32}, 
                                      A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                      S_ne::Ptr{Int32})::Cvoid
end

function bsc_import(::Type{Float32}, ::Type{Int64}, control, data, status, 
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_single_64.bsc_import(control::Ptr{bsc_control_type{Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         m::Int64, n::Int64, A_type::Ptr{Cchar},
                                         A_ne::Int64, A_row::Ptr{Int64}, 
                                         A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                         S_ne::Ptr{Int64})::Cvoid
end

function bsc_import(::Type{Float64}, ::Type{Int32}, control, data, status,
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_double.bsc_import(control::Ptr{bsc_control_type{Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                      m::Int32, n::Int32, A_type::Ptr{Cchar}, 
                                      A_ne::Int32, A_row::Ptr{Int32}, 
                                      A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                      S_ne::Ptr{Int32})::Cvoid
end

function bsc_import(::Type{Float64}, ::Type{Int64}, control, data, status,
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_double_64.bsc_import(control::Ptr{bsc_control_type{Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         m::Int64, n::Int64, A_type::Ptr{Cchar},
                                         A_ne::Int64, A_row::Ptr{Int64}, 
                                         A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                         S_ne::Ptr{Int64})::Cvoid
end

function bsc_import(::Type{Float128}, ::Type{Int32}, control, data, status,
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_quadruple.bsc_import(control::Ptr{bsc_control_type{Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         m::Int32, n::Int32, A_type::Ptr{Cchar},
                                         A_ne::Int32, A_row::Ptr{Int32}, 
                                         A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                         S_ne::Ptr{Int32})::Cvoid

end

function bsc_import(::Type{Float128}, ::Type{Int64}, control, data, status,
                    m, n, A_type, A_ne, A_row, A_col, A_ptr, S_ne)
  @ccall libgalahad_quadruple_64.bsc_import(control::Ptr{bsc_control_type{Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            m::Int64, n::Int64, A_type::Ptr{Cchar},
                                            A_ne::Int64, A_row::Ptr{Int64}, 
                                            A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                            S_ne::Ptr{Int64})::Cvoid
end

export bsc_reset_control

function bsc_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.bsc_reset_control(control::Ptr{bsc_control_type{Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function bsc_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.bsc_reset_control(control::Ptr{bsc_control_type{Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function bsc_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.bsc_reset_control(control::Ptr{bsc_control_type{Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function bsc_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.bsc_reset_control(control::Ptr{bsc_control_type{Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function bsc_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.bsc_reset_control(control::Ptr{bsc_control_type{Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function bsc_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.bsc_reset_control(control::Ptr{bsc_control_type{Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export bsc_form_s

function bsc_form_s(::Type{Float32}, ::Type{Int32}, data, status, m, n, 
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_single.bsc_form_s(data::Ptr{Ptr{Cvoid}}, 
                                      status::Ptr{Int32},
                                      m::Int32, n::Int32,
                                      a_ne::Int32, A_val::Ptr{Float32},
                                      S_ne::Int32, S_row::Ptr{Int32}, 
                                      S_col::Ptr{Int32}, S_ptr::Ptr{Int32},
                                      S_val::Ptr{Float32}, 
                                      D::Ptr{Float32})::Cvoid
end

function bsc_form_s(::Type{Float32}, ::Type{Int64}, data, status, m, n,
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_single_64.bsc_form_s(data::Ptr{Ptr{Cvoid}},
                                         status::Ptr{Int64},
                                         m::Int64, n::Int64,
                                         a_ne::Int64, A_val::Ptr{Float32},
                                         S_ne::Int64, S_row::Ptr{Int64}, 
                                         S_col::Ptr{Int64}, S_ptr::Ptr{Int64},
                                         S_val::Ptr{Float32}, 
                                         D::Ptr{Float32})::Cvoid
end

function bsc_form_s(::Type{Float64}, ::Type{Int32}, data, status, m, n,
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_double.bsc_form_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                      m::Int32, n::Int32,
                                      a_ne::Int32, A_val::Ptr{Float64},
                                      S_ne::Int32, S_row::Ptr{Int32}, 
                                      S_col::Ptr{Int32}, S_ptr::Ptr{Int32},
                                      S_val::Ptr{Float64}, 
                                      D::Ptr{Float64})::Cvoid
end

function bsc_form_s(::Type{Float64}, ::Type{Int64}, data, status, m, n,
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_double_64.bsc_form_s(data::Ptr{Ptr{Cvoid}}, 
                                         status::Ptr{Int64},
                                         m::Int64, n::Int64,
                                         a_ne::Int64, A_val::Ptr{Float64},
                                         S_ne::Int64, S_row::Ptr{Int64}, 
                                         S_col::Ptr{Int64}, S_ptr::Ptr{Int64},
                                         S_val::Ptr{Float64}, 
                                         D::Ptr{Float64})::Cvoid
end

function bsc_form_s(::Type{Float128}, ::Type{Int32}, data, status, m, n,
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_quadruple.bsc_form_s(data::Ptr{Ptr{Cvoid}}, 
                                         status::Ptr{Int32},
                                         m::Int32, n::Int32,
                                         a_ne::Int32, A_val::Ptr{Float128},
                                         S_ne::Int32, S_row::Ptr{Int32}, 
                                         S_col::Ptr{Int32}, S_ptr::Ptr{Int32},
                                         S_val::Ptr{Float128}, 
                                         D::Ptr{Float128})::Cvoid
end

function bsc_form_s(::Type{Float128}, ::Type{Int64}, data, status, m, n,
                    A_ne, A_val, S_ne, S_row, S_col, S_ptr, S_val, D)
  @ccall libgalahad_quadruple_64.bsc_form_s(data::Ptr{Ptr{Cvoid}}, 
                                            status::Ptr{Int64},
                                            m::Int64, n::Int64,
                                            a_ne::Int64, A_val::Ptr{Float128},
                                            S_ne::Int64, S_row::Ptr{Int64}, 
                                            S_col::Ptr{Int64}, S_ptr::Ptr{Int64},
                                            S_val::Ptr{Float128}, 
                                            D::Ptr{Float128})::Cvoid
end











