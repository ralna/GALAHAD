export rpd_control_type

struct rpd_control_type{INT}
  f_indexing::Bool
  qplib::INT
  error::INT
  out::INT
  print_level::INT
  space_critical::Bool
  deallocate_error_fatal::Bool
end

export rpd_inform_type

struct rpd_inform_type{INT}
  status::INT
  alloc_status::INT
  io_status::INT
  line::INT
  p_type::NTuple{4,Cchar}
  bad_alloc::NTuple{81,Cchar}
end

export rpd_initialize

function rpd_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{rpd_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function rpd_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{rpd_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function rpd_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{rpd_control_type{Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function rpd_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{rpd_control_type{Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function rpd_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{rpd_control_type{Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function rpd_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{rpd_control_type{Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export rpd_get_stats

function rpd_get_stats(::Type{Float32}, ::Type{Int32}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_single.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Int32,
                                         control::Ptr{rpd_control_type{Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         p_type::Ptr{Cchar}, n::Ptr{Int32}, m::Ptr{Int32},
                                         h_ne::Ptr{Int32}, a_ne::Ptr{Int32},
                                         h_c_ne::Ptr{Int32})::Cvoid
end

function rpd_get_stats(::Type{Float32}, ::Type{Int64}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_single_64.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Int64,
                                            control::Ptr{rpd_control_type{Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            p_type::Ptr{Cchar}, n::Ptr{Int64},
                                            m::Ptr{Int64}, h_ne::Ptr{Int64},
                                            a_ne::Ptr{Int64}, h_c_ne::Ptr{Int64})::Cvoid
end

function rpd_get_stats(::Type{Float64}, ::Type{Int32}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_double.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Int32,
                                         control::Ptr{rpd_control_type{Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         p_type::Ptr{Cchar}, n::Ptr{Int32}, m::Ptr{Int32},
                                         h_ne::Ptr{Int32}, a_ne::Ptr{Int32},
                                         h_c_ne::Ptr{Int32})::Cvoid
end

function rpd_get_stats(::Type{Float64}, ::Type{Int64}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_double_64.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Int64,
                                            control::Ptr{rpd_control_type{Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            p_type::Ptr{Cchar}, n::Ptr{Int64},
                                            m::Ptr{Int64}, h_ne::Ptr{Int64},
                                            a_ne::Ptr{Int64}, h_c_ne::Ptr{Int64})::Cvoid
end

function rpd_get_stats(::Type{Float128}, ::Type{Int32}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_quadruple.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Int32,
                                            control::Ptr{rpd_control_type{Int32}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                            p_type::Ptr{Cchar}, n::Ptr{Int32},
                                            m::Ptr{Int32}, h_ne::Ptr{Int32},
                                            a_ne::Ptr{Int32}, h_c_ne::Ptr{Int32})::Cvoid
end

function rpd_get_stats(::Type{Float128}, ::Type{Int64}, qplib_file, qplib_file_len, control,
                       data, status, p_type, n, m, h_ne, a_ne, h_c_ne)
  @ccall libgalahad_quadruple_64.rpd_get_stats(qplib_file::Ptr{Cchar},
                                               qplib_file_len::Int64,
                                               control::Ptr{rpd_control_type{Int64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                               p_type::Ptr{Cchar}, n::Ptr{Int64},
                                               m::Ptr{Int64}, h_ne::Ptr{Int64},
                                               a_ne::Ptr{Int64}, h_c_ne::Ptr{Int64})::Cvoid
end

export rpd_get_g

function rpd_get_g(::Type{Float32}, ::Type{Int32}, data, status, n, g)
  @ccall libgalahad_single.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     g::Ptr{Float32})::Cvoid
end

function rpd_get_g(::Type{Float32}, ::Type{Int64}, data, status, n, g)
  @ccall libgalahad_single_64.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        g::Ptr{Float32})::Cvoid
end

function rpd_get_g(::Type{Float64}, ::Type{Int32}, data, status, n, g)
  @ccall libgalahad_double.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     g::Ptr{Float64})::Cvoid
end

function rpd_get_g(::Type{Float64}, ::Type{Int64}, data, status, n, g)
  @ccall libgalahad_double_64.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        g::Ptr{Float64})::Cvoid
end

function rpd_get_g(::Type{Float128}, ::Type{Int32}, data, status, n, g)
  @ccall libgalahad_quadruple.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        g::Ptr{Float128})::Cvoid
end

function rpd_get_g(::Type{Float128}, ::Type{Int64}, data, status, n, g)
  @ccall libgalahad_quadruple_64.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, g::Ptr{Float128})::Cvoid
end

export rpd_get_f

function rpd_get_f(::Type{Float32}, ::Type{Int32}, data, status, f)
  @ccall libgalahad_single.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                     f::Ptr{Float32})::Cvoid
end

function rpd_get_f(::Type{Float32}, ::Type{Int64}, data, status, f)
  @ccall libgalahad_single_64.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        f::Ptr{Float32})::Cvoid
end

function rpd_get_f(::Type{Float64}, ::Type{Int32}, data, status, f)
  @ccall libgalahad_double.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                     f::Ptr{Float64})::Cvoid
end

function rpd_get_f(::Type{Float64}, ::Type{Int64}, data, status, f)
  @ccall libgalahad_double_64.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        f::Ptr{Float64})::Cvoid
end

function rpd_get_f(::Type{Float128}, ::Type{Int32}, data, status, f)
  @ccall libgalahad_quadruple.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                        f::Ptr{Float128})::Cvoid
end

function rpd_get_f(::Type{Float128}, ::Type{Int64}, data, status, f)
  @ccall libgalahad_quadruple_64.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           f::Ptr{Float128})::Cvoid
end

export rpd_get_xlu

function rpd_get_xlu(::Type{Float32}, ::Type{Int32}, data, status, n, x_l, x_u)
  @ccall libgalahad_single.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       x_l::Ptr{Float32}, x_u::Ptr{Float32})::Cvoid
end

function rpd_get_xlu(::Type{Float32}, ::Type{Int64}, data, status, n, x_l, x_u)
  @ccall libgalahad_single_64.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, x_l::Ptr{Float32},
                                          x_u::Ptr{Float32})::Cvoid
end

function rpd_get_xlu(::Type{Float64}, ::Type{Int32}, data, status, n, x_l, x_u)
  @ccall libgalahad_double.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

function rpd_get_xlu(::Type{Float64}, ::Type{Int64}, data, status, n, x_l, x_u)
  @ccall libgalahad_double_64.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64})::Cvoid
end

function rpd_get_xlu(::Type{Float128}, ::Type{Int32}, data, status, n, x_l, x_u)
  @ccall libgalahad_quadruple.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, x_l::Ptr{Float128},
                                          x_u::Ptr{Float128})::Cvoid
end

function rpd_get_xlu(::Type{Float128}, ::Type{Int64}, data, status, n, x_l, x_u)
  @ccall libgalahad_quadruple_64.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, x_l::Ptr{Float128},
                                             x_u::Ptr{Float128})::Cvoid
end

export rpd_get_clu

function rpd_get_clu(::Type{Float32}, ::Type{Int32}, data, status, m, c_l, c_u)
  @ccall libgalahad_single.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, m::Int32,
                                       c_l::Ptr{Float32}, c_u::Ptr{Float32})::Cvoid
end

function rpd_get_clu(::Type{Float32}, ::Type{Int64}, data, status, m, c_l, c_u)
  @ccall libgalahad_single_64.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          m::Int64, c_l::Ptr{Float32},
                                          c_u::Ptr{Float32})::Cvoid
end

function rpd_get_clu(::Type{Float64}, ::Type{Int32}, data, status, m, c_l, c_u)
  @ccall libgalahad_double.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, m::Int32,
                                       c_l::Ptr{Float64}, c_u::Ptr{Float64})::Cvoid
end

function rpd_get_clu(::Type{Float64}, ::Type{Int64}, data, status, m, c_l, c_u)
  @ccall libgalahad_double_64.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          m::Int64, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64})::Cvoid
end

function rpd_get_clu(::Type{Float128}, ::Type{Int32}, data, status, m, c_l, c_u)
  @ccall libgalahad_quadruple.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          m::Int32, c_l::Ptr{Float128},
                                          c_u::Ptr{Float128})::Cvoid
end

function rpd_get_clu(::Type{Float128}, ::Type{Int64}, data, status, m, c_l, c_u)
  @ccall libgalahad_quadruple_64.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             m::Int64, c_l::Ptr{Float128},
                                             c_u::Ptr{Float128})::Cvoid
end

export rpd_get_h

function rpd_get_h(::Type{Float32}, ::Type{Int32}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_single.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, h_ne::Int32,
                                     h_row::Ptr{Int32}, h_col::Ptr{Int32},
                                     h_val::Ptr{Float32})::Cvoid
end

function rpd_get_h(::Type{Float32}, ::Type{Int64}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_single_64.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        h_ne::Int64, h_row::Ptr{Int64}, h_col::Ptr{Int64},
                                        h_val::Ptr{Float32})::Cvoid
end

function rpd_get_h(::Type{Float64}, ::Type{Int32}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_double.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, h_ne::Int32,
                                     h_row::Ptr{Int32}, h_col::Ptr{Int32},
                                     h_val::Ptr{Float64})::Cvoid
end

function rpd_get_h(::Type{Float64}, ::Type{Int64}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_double_64.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        h_ne::Int64, h_row::Ptr{Int64}, h_col::Ptr{Int64},
                                        h_val::Ptr{Float64})::Cvoid
end

function rpd_get_h(::Type{Float128}, ::Type{Int32}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_quadruple.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                        h_ne::Int32, h_row::Ptr{Int32}, h_col::Ptr{Int32},
                                        h_val::Ptr{Float128})::Cvoid
end

function rpd_get_h(::Type{Float128}, ::Type{Int64}, data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_quadruple_64.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           h_ne::Int64, h_row::Ptr{Int64},
                                           h_col::Ptr{Int64}, h_val::Ptr{Float128})::Cvoid
end

export rpd_get_a

function rpd_get_a(::Type{Float32}, ::Type{Int32}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_single.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, a_ne::Int32,
                                     a_row::Ptr{Int32}, a_col::Ptr{Int32},
                                     a_val::Ptr{Float32})::Cvoid
end

function rpd_get_a(::Type{Float32}, ::Type{Int64}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_single_64.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        a_ne::Int64, a_row::Ptr{Int64}, a_col::Ptr{Int64},
                                        a_val::Ptr{Float32})::Cvoid
end

function rpd_get_a(::Type{Float64}, ::Type{Int32}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_double.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, a_ne::Int32,
                                     a_row::Ptr{Int32}, a_col::Ptr{Int32},
                                     a_val::Ptr{Float64})::Cvoid
end

function rpd_get_a(::Type{Float64}, ::Type{Int64}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_double_64.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                        a_ne::Int64, a_row::Ptr{Int64}, a_col::Ptr{Int64},
                                        a_val::Ptr{Float64})::Cvoid
end

function rpd_get_a(::Type{Float128}, ::Type{Int32}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_quadruple.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                        a_ne::Int32, a_row::Ptr{Int32}, a_col::Ptr{Int32},
                                        a_val::Ptr{Float128})::Cvoid
end

function rpd_get_a(::Type{Float128}, ::Type{Int64}, data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_quadruple_64.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           a_ne::Int64, a_row::Ptr{Int64},
                                           a_col::Ptr{Int64}, a_val::Ptr{Float128})::Cvoid
end

export rpd_get_h_c

function rpd_get_h_c(::Type{Float32}, ::Type{Int32}, data, status, h_c_ne, h_c_ptr, h_c_row,
                     h_c_col, h_c_val)
  @ccall libgalahad_single.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                       h_c_ne::Int32, h_c_ptr::Ptr{Int32},
                                       h_c_row::Ptr{Int32}, h_c_col::Ptr{Int32},
                                       h_c_val::Ptr{Float32})::Cvoid
end

function rpd_get_h_c(::Type{Float32}, ::Type{Int64}, data, status, h_c_ne, h_c_ptr, h_c_row,
                     h_c_col, h_c_val)
  @ccall libgalahad_single_64.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          h_c_ne::Int64, h_c_ptr::Ptr{Int64},
                                          h_c_row::Ptr{Int64}, h_c_col::Ptr{Int64},
                                          h_c_val::Ptr{Float32})::Cvoid
end

function rpd_get_h_c(::Type{Float64}, ::Type{Int32}, data, status, h_c_ne, h_c_ptr, h_c_row,
                     h_c_col, h_c_val)
  @ccall libgalahad_double.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                       h_c_ne::Int32, h_c_ptr::Ptr{Int32},
                                       h_c_row::Ptr{Int32}, h_c_col::Ptr{Int32},
                                       h_c_val::Ptr{Float64})::Cvoid
end

function rpd_get_h_c(::Type{Float64}, ::Type{Int64}, data, status, h_c_ne, h_c_ptr, h_c_row,
                     h_c_col, h_c_val)
  @ccall libgalahad_double_64.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          h_c_ne::Int64, h_c_ptr::Ptr{Int64},
                                          h_c_row::Ptr{Int64}, h_c_col::Ptr{Int64},
                                          h_c_val::Ptr{Float64})::Cvoid
end

function rpd_get_h_c(::Type{Float128}, ::Type{Int32}, data, status, h_c_ne, h_c_ptr,
                     h_c_row, h_c_col, h_c_val)
  @ccall libgalahad_quadruple.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          h_c_ne::Int32, h_c_ptr::Ptr{Int32},
                                          h_c_row::Ptr{Int32}, h_c_col::Ptr{Int32},
                                          h_c_val::Ptr{Float128})::Cvoid
end

function rpd_get_h_c(::Type{Float128}, ::Type{Int64}, data, status, h_c_ne, h_c_ptr,
                     h_c_row, h_c_col, h_c_val)
  @ccall libgalahad_quadruple_64.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             h_c_ne::Int64, h_c_ptr::Ptr{Int64},
                                             h_c_row::Ptr{Int64}, h_c_col::Ptr{Int64},
                                             h_c_val::Ptr{Float128})::Cvoid
end

export rpd_get_x_type

function rpd_get_x_type(::Type{Float32}, ::Type{Int32}, data, status, n, x_type)
  @ccall libgalahad_single.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, x_type::Ptr{Int32})::Cvoid
end

function rpd_get_x_type(::Type{Float32}, ::Type{Int64}, data, status, n, x_type)
  @ccall libgalahad_single_64.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, x_type::Ptr{Int64})::Cvoid
end

function rpd_get_x_type(::Type{Float64}, ::Type{Int32}, data, status, n, x_type)
  @ccall libgalahad_double.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, x_type::Ptr{Int32})::Cvoid
end

function rpd_get_x_type(::Type{Float64}, ::Type{Int64}, data, status, n, x_type)
  @ccall libgalahad_double_64.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, x_type::Ptr{Int64})::Cvoid
end

function rpd_get_x_type(::Type{Float128}, ::Type{Int32}, data, status, n, x_type)
  @ccall libgalahad_quadruple.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, x_type::Ptr{Int32})::Cvoid
end

function rpd_get_x_type(::Type{Float128}, ::Type{Int64}, data, status, n, x_type)
  @ccall libgalahad_quadruple_64.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, x_type::Ptr{Int64})::Cvoid
end

export rpd_get_x

function rpd_get_x(::Type{Float32}, ::Type{Int32}, data, status, n, x)
  @ccall libgalahad_single.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     x::Ptr{Float32})::Cvoid
end

function rpd_get_x(::Type{Float32}, ::Type{Int64}, data, status, n, x)
  @ccall libgalahad_single_64.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        x::Ptr{Float32})::Cvoid
end

function rpd_get_x(::Type{Float64}, ::Type{Int32}, data, status, n, x)
  @ccall libgalahad_double.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     x::Ptr{Float64})::Cvoid
end

function rpd_get_x(::Type{Float64}, ::Type{Int64}, data, status, n, x)
  @ccall libgalahad_double_64.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        x::Ptr{Float64})::Cvoid
end

function rpd_get_x(::Type{Float128}, ::Type{Int32}, data, status, n, x)
  @ccall libgalahad_quadruple.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        x::Ptr{Float128})::Cvoid
end

function rpd_get_x(::Type{Float128}, ::Type{Int64}, data, status, n, x)
  @ccall libgalahad_quadruple_64.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, x::Ptr{Float128})::Cvoid
end

export rpd_get_y

function rpd_get_y(::Type{Float32}, ::Type{Int32}, data, status, m, y)
  @ccall libgalahad_single.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, m::Int32,
                                     y::Ptr{Float32})::Cvoid
end

function rpd_get_y(::Type{Float32}, ::Type{Int64}, data, status, m, y)
  @ccall libgalahad_single_64.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, m::Int64,
                                        y::Ptr{Float32})::Cvoid
end

function rpd_get_y(::Type{Float64}, ::Type{Int32}, data, status, m, y)
  @ccall libgalahad_double.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, m::Int32,
                                     y::Ptr{Float64})::Cvoid
end

function rpd_get_y(::Type{Float64}, ::Type{Int64}, data, status, m, y)
  @ccall libgalahad_double_64.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, m::Int64,
                                        y::Ptr{Float64})::Cvoid
end

function rpd_get_y(::Type{Float128}, ::Type{Int32}, data, status, m, y)
  @ccall libgalahad_quadruple.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, m::Int32,
                                        y::Ptr{Float128})::Cvoid
end

function rpd_get_y(::Type{Float128}, ::Type{Int64}, data, status, m, y)
  @ccall libgalahad_quadruple_64.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           m::Int64, y::Ptr{Float128})::Cvoid
end

export rpd_get_z

function rpd_get_z(::Type{Float32}, ::Type{Int32}, data, status, n, z)
  @ccall libgalahad_single.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     z::Ptr{Float32})::Cvoid
end

function rpd_get_z(::Type{Float32}, ::Type{Int64}, data, status, n, z)
  @ccall libgalahad_single_64.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        z::Ptr{Float32})::Cvoid
end

function rpd_get_z(::Type{Float64}, ::Type{Int32}, data, status, n, z)
  @ccall libgalahad_double.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                     z::Ptr{Float64})::Cvoid
end

function rpd_get_z(::Type{Float64}, ::Type{Int64}, data, status, n, z)
  @ccall libgalahad_double_64.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64}, n::Int64,
                                        z::Ptr{Float64})::Cvoid
end

function rpd_get_z(::Type{Float128}, ::Type{Int32}, data, status, n, z)
  @ccall libgalahad_quadruple.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                        z::Ptr{Float128})::Cvoid
end

function rpd_get_z(::Type{Float128}, ::Type{Int64}, data, status, n, z)
  @ccall libgalahad_quadruple_64.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                           n::Int64, z::Ptr{Float128})::Cvoid
end

export rpd_information

function rpd_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.rpd_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{rpd_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function rpd_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.rpd_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{rpd_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function rpd_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.rpd_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{rpd_inform_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function rpd_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.rpd_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{rpd_inform_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function rpd_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.rpd_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{rpd_inform_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function rpd_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.rpd_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{rpd_inform_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export rpd_terminate

function rpd_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{rpd_control_type{Int32}},
                                         inform::Ptr{rpd_inform_type{Int32}})::Cvoid
end

function rpd_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{rpd_control_type{Int64}},
                                            inform::Ptr{rpd_inform_type{Int64}})::Cvoid
end

function rpd_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{rpd_control_type{Int32}},
                                         inform::Ptr{rpd_inform_type{Int32}})::Cvoid
end

function rpd_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{rpd_control_type{Int64}},
                                            inform::Ptr{rpd_inform_type{Int64}})::Cvoid
end

function rpd_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{rpd_control_type{Int32}},
                                            inform::Ptr{rpd_inform_type{Int32}})::Cvoid
end

function rpd_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{rpd_control_type{Int64}},
                                               inform::Ptr{rpd_inform_type{Int64}})::Cvoid
end
