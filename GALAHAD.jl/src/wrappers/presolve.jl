export presolve_control_type

struct presolve_control_type{T,INT}
  f_indexing::Bool
  termination::INT
  max_nbr_transforms::INT
  max_nbr_passes::INT
  c_accuracy::T
  z_accuracy::T
  infinity::T
  out::INT
  errout::INT
  print_level::INT
  dual_transformations::Bool
  redundant_xc::Bool
  primal_constraints_freq::INT
  dual_constraints_freq::INT
  singleton_columns_freq::INT
  doubleton_columns_freq::INT
  unc_variables_freq::INT
  dependent_variables_freq::INT
  sparsify_rows_freq::INT
  max_fill::INT
  transf_file_nbr::INT
  transf_buffer_size::INT
  transf_file_status::INT
  transf_file_name::NTuple{31,Cchar}
  y_sign::INT
  inactive_y::INT
  z_sign::INT
  inactive_z::INT
  final_x_bounds::INT
  final_z_bounds::INT
  final_c_bounds::INT
  final_y_bounds::INT
  check_primal_feasibility::INT
  check_dual_feasibility::INT
  pivot_tol::T
  min_rel_improve::T
  max_growth_factor::T
end

export presolve_inform_type

struct presolve_inform_type{INT}
  status::INT
  status_continue::INT
  status_continued::INT
  nbr_transforms::INT
  message::NTuple{3,NTuple{81,Cchar}}
end

export presolve_initialize

function presolve_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{presolve_control_type{Float32,
                                                                                  Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function presolve_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{presolve_control_type{Float32,
                                                                                     Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function presolve_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{presolve_control_type{Float64,
                                                                                  Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function presolve_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{presolve_control_type{Float64,
                                                                                     Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

function presolve_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                                  control::Ptr{presolve_control_type{Float128,
                                                                                     Int32}},
                                                  status::Ptr{Int32})::Cvoid
end

function presolve_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{presolve_control_type{Float128,
                                                                                        Int64}},
                                                     status::Ptr{Int64})::Cvoid
end

export presolve_read_specfile

function presolve_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.presolve_read_specfile(control::Ptr{presolve_control_type{Float32,
                                                                                     Int32}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

function presolve_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.presolve_read_specfile(control::Ptr{presolve_control_type{Float32,
                                                                                        Int64}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function presolve_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.presolve_read_specfile(control::Ptr{presolve_control_type{Float64,
                                                                                     Int32}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

function presolve_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.presolve_read_specfile(control::Ptr{presolve_control_type{Float64,
                                                                                        Int64}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function presolve_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.presolve_read_specfile(control::Ptr{presolve_control_type{Float128,
                                                                                        Int32}},
                                                     specfile::Ptr{Cchar})::Cvoid
end

function presolve_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.presolve_read_specfile(control::Ptr{presolve_control_type{Float128,
                                                                                           Int64}},
                                                        specfile::Ptr{Cchar})::Cvoid
end

export presolve_import_problem

function presolve_import_problem(::Type{Float32}, ::Type{Int32}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_single.presolve_import_problem(control::Ptr{presolve_control_type{Float32,
                                                                                      Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32, m::Int32,
                                                   H_type::Ptr{Cchar}, H_ne::Int32,
                                                   H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                                   H_ptr::Ptr{Int32}, H_val::Ptr{Float32},
                                                   g::Ptr{Float32}, f::Float32,
                                                   A_type::Ptr{Cchar}, A_ne::Int32,
                                                   A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                                   A_ptr::Ptr{Int32}, A_val::Ptr{Float32},
                                                   c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                                   x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                   n_out::Ptr{Int32}, m_out::Ptr{Int32},
                                                   H_ne_out::Ptr{Int32},
                                                   A_ne_out::Ptr{Int32})::Cvoid
end

function presolve_import_problem(::Type{Float32}, ::Type{Int64}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_single_64.presolve_import_problem(control::Ptr{presolve_control_type{Float32,
                                                                                         Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64,
                                                      m::Int64, H_type::Ptr{Cchar},
                                                      H_ne::Int64, H_row::Ptr{Int64},
                                                      H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                                      H_val::Ptr{Float32}, g::Ptr{Float32},
                                                      f::Float32, A_type::Ptr{Cchar},
                                                      A_ne::Int64, A_row::Ptr{Int64},
                                                      A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                      A_val::Ptr{Float32},
                                                      c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                                      x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                      n_out::Ptr{Int64}, m_out::Ptr{Int64},
                                                      H_ne_out::Ptr{Int64},
                                                      A_ne_out::Ptr{Int64})::Cvoid
end

function presolve_import_problem(::Type{Float64}, ::Type{Int32}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_double.presolve_import_problem(control::Ptr{presolve_control_type{Float64,
                                                                                      Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32, m::Int32,
                                                   H_type::Ptr{Cchar}, H_ne::Int32,
                                                   H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                                   H_ptr::Ptr{Int32}, H_val::Ptr{Float64},
                                                   g::Ptr{Float64}, f::Float64,
                                                   A_type::Ptr{Cchar}, A_ne::Int32,
                                                   A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                                   A_ptr::Ptr{Int32}, A_val::Ptr{Float64},
                                                   c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                   x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                   n_out::Ptr{Int32}, m_out::Ptr{Int32},
                                                   H_ne_out::Ptr{Int32},
                                                   A_ne_out::Ptr{Int32})::Cvoid
end

function presolve_import_problem(::Type{Float64}, ::Type{Int64}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_double_64.presolve_import_problem(control::Ptr{presolve_control_type{Float64,
                                                                                         Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64,
                                                      m::Int64, H_type::Ptr{Cchar},
                                                      H_ne::Int64, H_row::Ptr{Int64},
                                                      H_col::Ptr{Int64}, H_ptr::Ptr{Int64},
                                                      H_val::Ptr{Float64}, g::Ptr{Float64},
                                                      f::Float64, A_type::Ptr{Cchar},
                                                      A_ne::Int64, A_row::Ptr{Int64},
                                                      A_col::Ptr{Int64}, A_ptr::Ptr{Int64},
                                                      A_val::Ptr{Float64},
                                                      c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                      n_out::Ptr{Int64}, m_out::Ptr{Int64},
                                                      H_ne_out::Ptr{Int64},
                                                      A_ne_out::Ptr{Int64})::Cvoid
end

function presolve_import_problem(::Type{Float128}, ::Type{Int32}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_quadruple.presolve_import_problem(control::Ptr{presolve_control_type{Float128,
                                                                                         Int32}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, n::Int32,
                                                      m::Int32, H_type::Ptr{Cchar},
                                                      H_ne::Int32, H_row::Ptr{Int32},
                                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                      H_val::Ptr{Float128},
                                                      g::Ptr{Float128}, f::Cfloat128,
                                                      A_type::Ptr{Cchar}, A_ne::Int32,
                                                      A_row::Ptr{Int32}, A_col::Ptr{Int32},
                                                      A_ptr::Ptr{Int32},
                                                      A_val::Ptr{Float128},
                                                      c_l::Ptr{Float128},
                                                      c_u::Ptr{Float128},
                                                      x_l::Ptr{Float128},
                                                      x_u::Ptr{Float128}, n_out::Ptr{Int32},
                                                      m_out::Ptr{Int32},
                                                      H_ne_out::Ptr{Int32},
                                                      A_ne_out::Ptr{Int32})::Cvoid
end

function presolve_import_problem(::Type{Float128}, ::Type{Int64}, control, data, status, n,
                                 m, H_type, H_ne, H_row, H_col, H_ptr, H_val, g, f, A_type,
                                 A_ne, A_row, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u,
                                 n_out, m_out, H_ne_out, A_ne_out)
  @ccall libgalahad_quadruple_64.presolve_import_problem(control::Ptr{presolve_control_type{Float128,
                                                                                            Int64}},
                                                         data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, n::Int64,
                                                         m::Int64, H_type::Ptr{Cchar},
                                                         H_ne::Int64, H_row::Ptr{Int64},
                                                         H_col::Ptr{Int64},
                                                         H_ptr::Ptr{Int64},
                                                         H_val::Ptr{Float128},
                                                         g::Ptr{Float128}, f::Cfloat128,
                                                         A_type::Ptr{Cchar}, A_ne::Int64,
                                                         A_row::Ptr{Int64},
                                                         A_col::Ptr{Int64},
                                                         A_ptr::Ptr{Int64},
                                                         A_val::Ptr{Float128},
                                                         c_l::Ptr{Float128},
                                                         c_u::Ptr{Float128},
                                                         x_l::Ptr{Float128},
                                                         x_u::Ptr{Float128},
                                                         n_out::Ptr{Int64},
                                                         m_out::Ptr{Int64},
                                                         H_ne_out::Ptr{Int64},
                                                         A_ne_out::Ptr{Int64})::Cvoid
end

export presolve_transform_problem

function presolve_transform_problem(::Type{Float32}, ::Type{Int32}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_single.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, n::Int32,
                                                      m::Int32, H_ne::Int32,
                                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                      H_val::Ptr{Float32}, g::Ptr{Float32},
                                                      f::Ptr{Float32}, A_ne::Int32,
                                                      A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                      A_val::Ptr{Float32},
                                                      c_l::Ptr{Float32}, c_u::Ptr{Float32},
                                                      x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                      y_l::Ptr{Float32}, y_u::Ptr{Float32},
                                                      z_l::Ptr{Float32},
                                                      z_u::Ptr{Float32})::Cvoid
end

function presolve_transform_problem(::Type{Float32}, ::Type{Int64}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_single_64.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, n::Int64,
                                                         m::Int64, H_ne::Int64,
                                                         H_col::Ptr{Int64},
                                                         H_ptr::Ptr{Int64},
                                                         H_val::Ptr{Float32},
                                                         g::Ptr{Float32}, f::Ptr{Float32},
                                                         A_ne::Int64, A_col::Ptr{Int64},
                                                         A_ptr::Ptr{Int64},
                                                         A_val::Ptr{Float32},
                                                         c_l::Ptr{Float32},
                                                         c_u::Ptr{Float32},
                                                         x_l::Ptr{Float32},
                                                         x_u::Ptr{Float32},
                                                         y_l::Ptr{Float32},
                                                         y_u::Ptr{Float32},
                                                         z_l::Ptr{Float32},
                                                         z_u::Ptr{Float32})::Cvoid
end

function presolve_transform_problem(::Type{Float64}, ::Type{Int32}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_double.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int32}, n::Int32,
                                                      m::Int32, H_ne::Int32,
                                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32},
                                                      H_val::Ptr{Float64}, g::Ptr{Float64},
                                                      f::Ptr{Float64}, A_ne::Int32,
                                                      A_col::Ptr{Int32}, A_ptr::Ptr{Int32},
                                                      A_val::Ptr{Float64},
                                                      c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                      x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                      y_l::Ptr{Float64}, y_u::Ptr{Float64},
                                                      z_l::Ptr{Float64},
                                                      z_u::Ptr{Float64})::Cvoid
end

function presolve_transform_problem(::Type{Float64}, ::Type{Int64}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_double_64.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int64}, n::Int64,
                                                         m::Int64, H_ne::Int64,
                                                         H_col::Ptr{Int64},
                                                         H_ptr::Ptr{Int64},
                                                         H_val::Ptr{Float64},
                                                         g::Ptr{Float64}, f::Ptr{Float64},
                                                         A_ne::Int64, A_col::Ptr{Int64},
                                                         A_ptr::Ptr{Int64},
                                                         A_val::Ptr{Float64},
                                                         c_l::Ptr{Float64},
                                                         c_u::Ptr{Float64},
                                                         x_l::Ptr{Float64},
                                                         x_u::Ptr{Float64},
                                                         y_l::Ptr{Float64},
                                                         y_u::Ptr{Float64},
                                                         z_l::Ptr{Float64},
                                                         z_u::Ptr{Float64})::Cvoid
end

function presolve_transform_problem(::Type{Float128}, ::Type{Int32}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_quadruple.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                         status::Ptr{Int32}, n::Int32,
                                                         m::Int32, H_ne::Int32,
                                                         H_col::Ptr{Int32},
                                                         H_ptr::Ptr{Int32},
                                                         H_val::Ptr{Float128},
                                                         g::Ptr{Float128}, f::Ptr{Float128},
                                                         A_ne::Int32, A_col::Ptr{Int32},
                                                         A_ptr::Ptr{Int32},
                                                         A_val::Ptr{Float128},
                                                         c_l::Ptr{Float128},
                                                         c_u::Ptr{Float128},
                                                         x_l::Ptr{Float128},
                                                         x_u::Ptr{Float128},
                                                         y_l::Ptr{Float128},
                                                         y_u::Ptr{Float128},
                                                         z_l::Ptr{Float128},
                                                         z_u::Ptr{Float128})::Cvoid
end

function presolve_transform_problem(::Type{Float128}, ::Type{Int64}, data, status, n, m,
                                    H_ne, H_col, H_ptr, H_val, g, f, A_ne, A_col, A_ptr,
                                    A_val, c_l, c_u, x_l, x_u, y_l, y_u, z_l, z_u)
  @ccall libgalahad_quadruple_64.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                            status::Ptr{Int64}, n::Int64,
                                                            m::Int64, H_ne::Int64,
                                                            H_col::Ptr{Int64},
                                                            H_ptr::Ptr{Int64},
                                                            H_val::Ptr{Float128},
                                                            g::Ptr{Float128},
                                                            f::Ptr{Float128}, A_ne::Int64,
                                                            A_col::Ptr{Int64},
                                                            A_ptr::Ptr{Int64},
                                                            A_val::Ptr{Float128},
                                                            c_l::Ptr{Float128},
                                                            c_u::Ptr{Float128},
                                                            x_l::Ptr{Float128},
                                                            x_u::Ptr{Float128},
                                                            y_l::Ptr{Float128},
                                                            y_u::Ptr{Float128},
                                                            z_l::Ptr{Float128},
                                                            z_u::Ptr{Float128})::Cvoid
end

export presolve_restore_solution

function presolve_restore_solution(::Type{Float32}, ::Type{Int32}, data, status, n_in, m_in,
                                   x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_single.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n_in::Int32,
                                                     m_in::Int32, x_in::Ptr{Float32},
                                                     c_in::Ptr{Float32}, y_in::Ptr{Float32},
                                                     z_in::Ptr{Float32}, n::Int32, m::Int32,
                                                     x::Ptr{Float32}, c::Ptr{Float32},
                                                     y::Ptr{Float32},
                                                     z::Ptr{Float32})::Cvoid
end

function presolve_restore_solution(::Type{Float32}, ::Type{Int64}, data, status, n_in, m_in,
                                   x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_single_64.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n_in::Int64,
                                                        m_in::Int64, x_in::Ptr{Float32},
                                                        c_in::Ptr{Float32},
                                                        y_in::Ptr{Float32},
                                                        z_in::Ptr{Float32}, n::Int64,
                                                        m::Int64, x::Ptr{Float32},
                                                        c::Ptr{Float32}, y::Ptr{Float32},
                                                        z::Ptr{Float32})::Cvoid
end

function presolve_restore_solution(::Type{Float64}, ::Type{Int32}, data, status, n_in, m_in,
                                   x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_double.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32}, n_in::Int32,
                                                     m_in::Int32, x_in::Ptr{Float64},
                                                     c_in::Ptr{Float64}, y_in::Ptr{Float64},
                                                     z_in::Ptr{Float64}, n::Int32, m::Int32,
                                                     x::Ptr{Float64}, c::Ptr{Float64},
                                                     y::Ptr{Float64},
                                                     z::Ptr{Float64})::Cvoid
end

function presolve_restore_solution(::Type{Float64}, ::Type{Int64}, data, status, n_in, m_in,
                                   x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_double_64.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64}, n_in::Int64,
                                                        m_in::Int64, x_in::Ptr{Float64},
                                                        c_in::Ptr{Float64},
                                                        y_in::Ptr{Float64},
                                                        z_in::Ptr{Float64}, n::Int64,
                                                        m::Int64, x::Ptr{Float64},
                                                        c::Ptr{Float64}, y::Ptr{Float64},
                                                        z::Ptr{Float64})::Cvoid
end

function presolve_restore_solution(::Type{Float128}, ::Type{Int32}, data, status, n_in,
                                   m_in, x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_quadruple.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32}, n_in::Int32,
                                                        m_in::Int32, x_in::Ptr{Float128},
                                                        c_in::Ptr{Float128},
                                                        y_in::Ptr{Float128},
                                                        z_in::Ptr{Float128}, n::Int32,
                                                        m::Int32, x::Ptr{Float128},
                                                        c::Ptr{Float128}, y::Ptr{Float128},
                                                        z::Ptr{Float128})::Cvoid
end

function presolve_restore_solution(::Type{Float128}, ::Type{Int64}, data, status, n_in,
                                   m_in, x_in, c_in, y_in, z_in, n, m, x, c, y, z)
  @ccall libgalahad_quadruple_64.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int64}, n_in::Int64,
                                                           m_in::Int64, x_in::Ptr{Float128},
                                                           c_in::Ptr{Float128},
                                                           y_in::Ptr{Float128},
                                                           z_in::Ptr{Float128}, n::Int64,
                                                           m::Int64, x::Ptr{Float128},
                                                           c::Ptr{Float128},
                                                           y::Ptr{Float128},
                                                           z::Ptr{Float128})::Cvoid
end

export presolve_information

function presolve_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{presolve_inform_type{Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function presolve_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{presolve_inform_type{Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function presolve_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                inform::Ptr{presolve_inform_type{Int32}},
                                                status::Ptr{Int32})::Cvoid
end

function presolve_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{presolve_inform_type{Int64}},
                                                   status::Ptr{Int64})::Cvoid
end

function presolve_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                   inform::Ptr{presolve_inform_type{Int32}},
                                                   status::Ptr{Int32})::Cvoid
end

function presolve_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                      inform::Ptr{presolve_inform_type{Int64}},
                                                      status::Ptr{Int64})::Cvoid
end

export presolve_terminate

function presolve_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{presolve_control_type{Float32,
                                                                                 Int32}},
                                              inform::Ptr{presolve_inform_type{Int32}})::Cvoid
end

function presolve_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{presolve_control_type{Float32,
                                                                                    Int64}},
                                                 inform::Ptr{presolve_inform_type{Int64}})::Cvoid
end

function presolve_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{presolve_control_type{Float64,
                                                                                 Int32}},
                                              inform::Ptr{presolve_inform_type{Int32}})::Cvoid
end

function presolve_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{presolve_control_type{Float64,
                                                                                    Int64}},
                                                 inform::Ptr{presolve_inform_type{Int64}})::Cvoid
end

function presolve_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{presolve_control_type{Float128,
                                                                                    Int32}},
                                                 inform::Ptr{presolve_inform_type{Int32}})::Cvoid
end

function presolve_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{presolve_control_type{Float128,
                                                                                       Int64}},
                                                    inform::Ptr{presolve_inform_type{Int64}})::Cvoid
end
