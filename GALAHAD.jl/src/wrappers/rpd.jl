export rpd_control_type

mutable struct rpd_control_type
  f_indexing::Bool
  qplib::Cint
  error::Cint
  out::Cint
  print_level::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool

  rpd_control_type() = new()
end

export rpd_inform_type

mutable struct rpd_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
  io_status::Cint
  line::Cint
  p_type::NTuple{4,Cchar}

  rpd_inform_type() = new()
end

export rpd_initialize_s

function rpd_initialize_s(data, control, status)
  @ccall libgalahad_single.rpd_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{rpd_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export rpd_initialize

function rpd_initialize(data, control, status)
  @ccall libgalahad_double.rpd_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{rpd_control_type},
                                          status::Ptr{Cint})::Cvoid
end

export rpd_get_stats_s

function rpd_get_stats_s(qplib_file, qplib_file_len, control, data, status, p_type, n, m,
                       h_ne, a_ne, h_c_ne)
  @ccall libgalahad_single.rpd_get_stats_s(qplib_file::Ptr{Cchar}, qplib_file_len::Cint,
                                           control::Ref{rpd_control_type},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           p_type::Ptr{Cchar}, n::Ptr{Cint}, m::Ptr{Cint},
                                           h_ne::Ptr{Cint}, a_ne::Ptr{Cint},
                                           h_c_ne::Ptr{Cint})::Cvoid
end

export rpd_get_stats

function rpd_get_stats(qplib_file, qplib_file_len, control, data, status, p_type, n, m,
                     h_ne, a_ne, h_c_ne)
  @ccall libgalahad_double.rpd_get_stats(qplib_file::Ptr{Cchar}, qplib_file_len::Cint,
                                         control::Ref{rpd_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                         p_type::Ptr{Cchar}, n::Ptr{Cint}, m::Ptr{Cint},
                                         h_ne::Ptr{Cint}, a_ne::Ptr{Cint},
                                         h_c_ne::Ptr{Cint})::Cvoid
end

export rpd_get_g_s

function rpd_get_g_s(data, status, n, g)
  @ccall libgalahad_single.rpd_get_g_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       g::Ptr{Float32})::Cvoid
end

export rpd_get_g

function rpd_get_g(data, status, n, g)
  @ccall libgalahad_double.rpd_get_g(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                     g::Ptr{Float64})::Cvoid
end

export rpd_get_f_s

function rpd_get_f_s(data, status, f)
  @ccall libgalahad_single.rpd_get_f_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                       f::Ptr{Float32})::Cvoid
end

export rpd_get_f

function rpd_get_f(data, status, f)
  @ccall libgalahad_double.rpd_get_f(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                     f::Ptr{Float64})::Cvoid
end

export rpd_get_xlu_s

function rpd_get_xlu_s(data, status, n, x_l, x_u)
  @ccall libgalahad_single.rpd_get_xlu_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         x_l::Ptr{Float32}, x_u::Ptr{Float32})::Cvoid
end

export rpd_get_xlu

function rpd_get_xlu(data, status, n, x_l, x_u)
  @ccall libgalahad_double.rpd_get_xlu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

export rpd_get_clu_s

function rpd_get_clu_s(data, status, m, c_l, c_u)
  @ccall libgalahad_single.rpd_get_clu_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                         c_l::Ptr{Float32}, c_u::Ptr{Float32})::Cvoid
end

export rpd_get_clu

function rpd_get_clu(data, status, m, c_l, c_u)
  @ccall libgalahad_double.rpd_get_clu(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       c_l::Ptr{Float64}, c_u::Ptr{Float64})::Cvoid
end

export rpd_get_h_s

function rpd_get_h_s(data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_single.rpd_get_h_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, h_ne::Cint,
                                       h_row::Ptr{Cint}, h_col::Ptr{Cint},
                                       h_val::Ptr{Float32})::Cvoid
end

export rpd_get_h

function rpd_get_h(data, status, h_ne, h_row, h_col, h_val)
  @ccall libgalahad_double.rpd_get_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, h_ne::Cint,
                                     h_row::Ptr{Cint}, h_col::Ptr{Cint},
                                     h_val::Ptr{Float64})::Cvoid
end

export rpd_get_a_s

function rpd_get_a_s(data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_single.rpd_get_a_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, a_ne::Cint,
                                       a_row::Ptr{Cint}, a_col::Ptr{Cint},
                                       a_val::Ptr{Float32})::Cvoid
end

export rpd_get_a

function rpd_get_a(data, status, a_ne, a_row, a_col, a_val)
  @ccall libgalahad_double.rpd_get_a(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, a_ne::Cint,
                                     a_row::Ptr{Cint}, a_col::Ptr{Cint},
                                     a_val::Ptr{Float64})::Cvoid
end

export rpd_get_h_c_s

function rpd_get_h_c_s(data, status, h_c_ne, h_c_ptr, h_c_row, h_c_col, h_c_val)
  @ccall libgalahad_single.rpd_get_h_c_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                         h_c_ne::Cint, h_c_ptr::Ptr{Cint},
                                         h_c_row::Ptr{Cint}, h_c_col::Ptr{Cint},
                                         h_c_val::Ptr{Float32})::Cvoid
end

export rpd_get_h_c

function rpd_get_h_c(data, status, h_c_ne, h_c_ptr, h_c_row, h_c_col, h_c_val)
  @ccall libgalahad_double.rpd_get_h_c(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                       h_c_ne::Cint, h_c_ptr::Ptr{Cint},
                                       h_c_row::Ptr{Cint}, h_c_col::Ptr{Cint},
                                       h_c_val::Ptr{Float64})::Cvoid
end

export rpd_get_x_type_s

function rpd_get_x_type_s(data, status, n, x_type)
  @ccall libgalahad_single.rpd_get_x_type_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, x_type::Ptr{Cint})::Cvoid
end

export rpd_get_x_type

function rpd_get_x_type(data, status, n, x_type)
  @ccall libgalahad_double.rpd_get_x_type(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                          n::Cint, x_type::Ptr{Cint})::Cvoid
end

export rpd_get_x_s

function rpd_get_x_s(data, status, n, x)
  @ccall libgalahad_single.rpd_get_x_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       x::Ptr{Float32})::Cvoid
end

export rpd_get_x

function rpd_get_x(data, status, n, x)
  @ccall libgalahad_double.rpd_get_x(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                     x::Ptr{Float64})::Cvoid
end

export rpd_get_y_s

function rpd_get_y_s(data, status, m, y)
  @ccall libgalahad_single.rpd_get_y_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                       y::Ptr{Float32})::Cvoid
end

export rpd_get_y

function rpd_get_y(data, status, m, y)
  @ccall libgalahad_double.rpd_get_y(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, m::Cint,
                                     y::Ptr{Float64})::Cvoid
end

export rpd_get_z_s

function rpd_get_z_s(data, status, n, z)
  @ccall libgalahad_single.rpd_get_z_s(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       z::Ptr{Float32})::Cvoid
end

export rpd_get_z

function rpd_get_z(data, status, n, z)
  @ccall libgalahad_double.rpd_get_z(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                     z::Ptr{Float64})::Cvoid
end

export rpd_information_s

function rpd_information_s(data, inform, status)
  @ccall libgalahad_single.rpd_information_s(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{rpd_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export rpd_information

function rpd_information(data, inform, status)
  @ccall libgalahad_double.rpd_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ref{rpd_inform_type},
                                           status::Ptr{Cint})::Cvoid
end

export rpd_terminate_s

function rpd_terminate_s(data, control, inform)
  @ccall libgalahad_single.rpd_terminate_s(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{rpd_control_type},
                                           inform::Ref{rpd_inform_type})::Cvoid
end

export rpd_terminate

function rpd_terminate(data, control, inform)
  @ccall libgalahad_double.rpd_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ref{rpd_control_type},
                                         inform::Ref{rpd_inform_type})::Cvoid
end
