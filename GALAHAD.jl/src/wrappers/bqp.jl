export bqp_control_type

struct bqp_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  cold_start::INT
  ratio_cg_vs_sd::INT
  change_max::INT
  cg_maxit::INT
  sif_file_device::INT
  infinity::T
  stop_p::T
  stop_d::T
  stop_c::T
  identical_bounds_tol::T
  stop_cg_relative::T
  stop_cg_absolute::T
  zero_curvature::T
  cpu_time_limit::T
  exact_arcsearch::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T,INT}
end

export bqp_time_type

struct bqp_time_type
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
end

export bqp_inform_type

struct bqp_inform_type{T,INT}
  status::INT
  alloc_status::INT
  factorization_status::INT
  iter::INT
  cg_iter::INT
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::bqp_time_type
  sbls_inform::sbls_inform_type{T,INT}
end

export bqp_initialize

function bqp_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bqp_control_type{Float32,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function bqp_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bqp_control_type{Float32,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function bqp_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{bqp_control_type{Float64,Int32}},
                                          status::Ptr{Int32})::Cvoid
end

function bqp_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bqp_control_type{Float64,Int64}},
                                             status::Ptr{Int64})::Cvoid
end

function bqp_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{bqp_control_type{Float128,Int32}},
                                             status::Ptr{Int32})::Cvoid
end

function bqp_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{bqp_control_type{Float128,
                                                                              Int64}},
                                                status::Ptr{Int64})::Cvoid
end

export bqp_read_specfile

function bqp_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.bqp_read_specfile(control::Ptr{bqp_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function bqp_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.bqp_read_specfile(control::Ptr{bqp_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function bqp_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.bqp_read_specfile(control::Ptr{bqp_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function bqp_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.bqp_read_specfile(control::Ptr{bqp_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function bqp_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.bqp_read_specfile(control::Ptr{bqp_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function bqp_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.bqp_read_specfile(control::Ptr{bqp_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export bqp_import

function bqp_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single.bqp_import(control::Ptr{bqp_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function bqp_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_single_64.bqp_import(control::Ptr{bqp_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function bqp_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double.bqp_import(control::Ptr{bqp_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                      H_type::Ptr{Cchar}, ne::Int32, H_row::Ptr{Int32},
                                      H_col::Ptr{Int32}, H_ptr::Ptr{Int32})::Cvoid
end

function bqp_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_double_64.bqp_import(control::Ptr{bqp_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                         n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                         H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                         H_ptr::Ptr{Int64})::Cvoid
end

function bqp_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple.bqp_import(control::Ptr{bqp_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                         n::Int32, H_type::Ptr{Cchar}, ne::Int32,
                                         H_row::Ptr{Int32}, H_col::Ptr{Int32},
                                         H_ptr::Ptr{Int32})::Cvoid
end

function bqp_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, H_type, ne,
                    H_row, H_col, H_ptr)
  @ccall libgalahad_quadruple_64.bqp_import(control::Ptr{bqp_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                            n::Int64, H_type::Ptr{Cchar}, ne::Int64,
                                            H_row::Ptr{Int64}, H_col::Ptr{Int64},
                                            H_ptr::Ptr{Int64})::Cvoid
end

export bqp_import_without_h

function bqp_import_without_h(::Type{Float32}, ::Type{Int32}, control, data, status, n)
  @ccall libgalahad_single.bqp_import_without_h(control::Ptr{bqp_control_type{Float32,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32)::Cvoid
end

function bqp_import_without_h(::Type{Float32}, ::Type{Int64}, control, data, status, n)
  @ccall libgalahad_single_64.bqp_import_without_h(control::Ptr{bqp_control_type{Float32,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64)::Cvoid
end

function bqp_import_without_h(::Type{Float64}, ::Type{Int32}, control, data, status, n)
  @ccall libgalahad_double.bqp_import_without_h(control::Ptr{bqp_control_type{Float64,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32)::Cvoid
end

function bqp_import_without_h(::Type{Float64}, ::Type{Int64}, control, data, status, n)
  @ccall libgalahad_double_64.bqp_import_without_h(control::Ptr{bqp_control_type{Float64,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64)::Cvoid
end

function bqp_import_without_h(::Type{Float128}, ::Type{Int32}, control, data, status, n)
  @ccall libgalahad_quadruple.bqp_import_without_h(control::Ptr{bqp_control_type{Float128,
                                                                                 Int32}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int32}, n::Int32)::Cvoid
end

function bqp_import_without_h(::Type{Float128}, ::Type{Int64}, control, data, status, n)
  @ccall libgalahad_quadruple_64.bqp_import_without_h(control::Ptr{bqp_control_type{Float128,
                                                                                    Int64}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Int64}, n::Int64)::Cvoid
end

export bqp_reset_control

function bqp_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.bqp_reset_control(control::Ptr{bqp_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function bqp_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.bqp_reset_control(control::Ptr{bqp_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function bqp_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.bqp_reset_control(control::Ptr{bqp_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function bqp_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.bqp_reset_control(control::Ptr{bqp_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function bqp_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.bqp_reset_control(control::Ptr{bqp_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function bqp_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.bqp_reset_control(control::Ptr{bqp_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export bqp_solve_given_h

function bqp_solve_given_h(::Type{Float32}, ::Type{Int32}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_single.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, h_ne::Int32, H_val::Ptr{Float32},
                                             g::Ptr{Float32}, f::Float32, x_l::Ptr{Float32},
                                             x_u::Ptr{Float32}, x::Ptr{Float32},
                                             z::Ptr{Float32}, x_stat::Ptr{Int32})::Cvoid
end

function bqp_solve_given_h(::Type{Float32}, ::Type{Int64}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_single_64.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, h_ne::Int64, H_val::Ptr{Float32},
                                                g::Ptr{Float32}, f::Float32,
                                                x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                x::Ptr{Float32}, z::Ptr{Float32},
                                                x_stat::Ptr{Int64})::Cvoid
end

function bqp_solve_given_h(::Type{Float64}, ::Type{Int32}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_double.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                             n::Int32, h_ne::Int32, H_val::Ptr{Float64},
                                             g::Ptr{Float64}, f::Float64, x_l::Ptr{Float64},
                                             x_u::Ptr{Float64}, x::Ptr{Float64},
                                             z::Ptr{Float64}, x_stat::Ptr{Int32})::Cvoid
end

function bqp_solve_given_h(::Type{Float64}, ::Type{Int64}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_double_64.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                                n::Int64, h_ne::Int64, H_val::Ptr{Float64},
                                                g::Ptr{Float64}, f::Float64,
                                                x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                x::Ptr{Float64}, z::Ptr{Float64},
                                                x_stat::Ptr{Int64})::Cvoid
end

function bqp_solve_given_h(::Type{Float128}, ::Type{Int32}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_quadruple.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                n::Int32, h_ne::Int32, H_val::Ptr{Float128},
                                                g::Ptr{Float128}, f::Cfloat128,
                                                x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                                x::Ptr{Float128}, z::Ptr{Float128},
                                                x_stat::Ptr{Int32})::Cvoid
end

function bqp_solve_given_h(::Type{Float128}, ::Type{Int64}, data, status, n, h_ne, H_val, g,
                           f, x_l, x_u, x, z, x_stat)
  @ccall libgalahad_quadruple_64.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64}, n::Int64,
                                                   h_ne::Int64, H_val::Ptr{Float128},
                                                   g::Ptr{Float128}, f::Cfloat128,
                                                   x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                                   x::Ptr{Float128}, z::Ptr{Float128},
                                                   x_stat::Ptr{Int64})::Cvoid
end

export bqp_solve_reverse_h_prod

function bqp_solve_reverse_h_prod(::Type{Float32}, ::Type{Int32}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_single.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    g::Ptr{Float32}, f::Float32,
                                                    x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                    x::Ptr{Float32}, z::Ptr{Float32},
                                                    x_stat::Ptr{Int32}, v::Ptr{Float32},
                                                    prod::Ptr{Float32}, nz_v::Ptr{Int32},
                                                    nz_v_start::Ptr{Int32},
                                                    nz_v_end::Ptr{Int32},
                                                    nz_prod::Ptr{Int32},
                                                    nz_prod_end::Int32)::Cvoid
end

function bqp_solve_reverse_h_prod(::Type{Float32}, ::Type{Int64}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_single_64.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       g::Ptr{Float32}, f::Float32,
                                                       x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                       x::Ptr{Float32}, z::Ptr{Float32},
                                                       x_stat::Ptr{Int64}, v::Ptr{Float32},
                                                       prod::Ptr{Float32}, nz_v::Ptr{Int64},
                                                       nz_v_start::Ptr{Int64},
                                                       nz_v_end::Ptr{Int64},
                                                       nz_prod::Ptr{Int64},
                                                       nz_prod_end::Int64)::Cvoid
end

function bqp_solve_reverse_h_prod(::Type{Float64}, ::Type{Int32}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_double.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    g::Ptr{Float64}, f::Float64,
                                                    x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                    x::Ptr{Float64}, z::Ptr{Float64},
                                                    x_stat::Ptr{Int32}, v::Ptr{Float64},
                                                    prod::Ptr{Float64}, nz_v::Ptr{Int32},
                                                    nz_v_start::Ptr{Int32},
                                                    nz_v_end::Ptr{Int32},
                                                    nz_prod::Ptr{Int32},
                                                    nz_prod_end::Int32)::Cvoid
end

function bqp_solve_reverse_h_prod(::Type{Float64}, ::Type{Int64}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_double_64.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       g::Ptr{Float64}, f::Float64,
                                                       x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                       x::Ptr{Float64}, z::Ptr{Float64},
                                                       x_stat::Ptr{Int64}, v::Ptr{Float64},
                                                       prod::Ptr{Float64}, nz_v::Ptr{Int64},
                                                       nz_v_start::Ptr{Int64},
                                                       nz_v_end::Ptr{Int64},
                                                       nz_prod::Ptr{Int64},
                                                       nz_prod_end::Int64)::Cvoid
end

function bqp_solve_reverse_h_prod(::Type{Float128}, ::Type{Int32}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_quadruple.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int32}, n::Int32,
                                                       g::Ptr{Float128}, f::Cfloat128,
                                                       x_l::Ptr{Float128},
                                                       x_u::Ptr{Float128}, x::Ptr{Float128},
                                                       z::Ptr{Float128}, x_stat::Ptr{Int32},
                                                       v::Ptr{Float128},
                                                       prod::Ptr{Float128},
                                                       nz_v::Ptr{Int32},
                                                       nz_v_start::Ptr{Int32},
                                                       nz_v_end::Ptr{Int32},
                                                       nz_prod::Ptr{Int32},
                                                       nz_prod_end::Int32)::Cvoid
end

function bqp_solve_reverse_h_prod(::Type{Float128}, ::Type{Int64}, data, status, n, g, f,
                                  x_l, x_u, x, z, x_stat, v, prod, nz_v, nz_v_start,
                                  nz_v_end, nz_prod, nz_prod_end)
  @ccall libgalahad_quadruple_64.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Int64}, n::Int64,
                                                          g::Ptr{Float128}, f::Cfloat128,
                                                          x_l::Ptr{Float128},
                                                          x_u::Ptr{Float128},
                                                          x::Ptr{Float128},
                                                          z::Ptr{Float128},
                                                          x_stat::Ptr{Int64},
                                                          v::Ptr{Float128},
                                                          prod::Ptr{Float128},
                                                          nz_v::Ptr{Int64},
                                                          nz_v_start::Ptr{Int64},
                                                          nz_v_end::Ptr{Int64},
                                                          nz_prod::Ptr{Int64},
                                                          nz_prod_end::Int64)::Cvoid
end

export bqp_information

function bqp_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.bqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bqp_inform_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bqp_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.bqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bqp_inform_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bqp_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.bqp_information(data::Ptr{Ptr{Cvoid}},
                                           inform::Ptr{bqp_inform_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function bqp_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.bqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bqp_inform_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function bqp_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.bqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{bqp_inform_type{Float128,Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function bqp_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.bqp_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{bqp_inform_type{Float128,
                                                                             Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export bqp_terminate

function bqp_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bqp_control_type{Float32,Int32}},
                                         inform::Ptr{bqp_inform_type{Float32,Int32}})::Cvoid
end

function bqp_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bqp_control_type{Float32,Int64}},
                                            inform::Ptr{bqp_inform_type{Float32,Int64}})::Cvoid
end

function bqp_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                         control::Ptr{bqp_control_type{Float64,Int32}},
                                         inform::Ptr{bqp_inform_type{Float64,Int32}})::Cvoid
end

function bqp_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bqp_control_type{Float64,Int64}},
                                            inform::Ptr{bqp_inform_type{Float64,Int64}})::Cvoid
end

function bqp_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bqp_control_type{Float128,Int32}},
                                            inform::Ptr{bqp_inform_type{Float128,Int32}})::Cvoid
end

function bqp_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{bqp_control_type{Float128,
                                                                             Int64}},
                                               inform::Ptr{bqp_inform_type{Float128,Int64}})::Cvoid
end
