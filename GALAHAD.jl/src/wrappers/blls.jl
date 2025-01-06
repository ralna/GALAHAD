export blls_control_type

struct blls_control_type{T,INT}
  f_indexing::Bool
  error::INT
  out::INT
  print_level::INT
  start_print::INT
  stop_print::INT
  print_gap::INT
  maxit::INT
  cold_start::INT
  preconditioner::INT
  ratio_cg_vs_sd::INT
  change_max::INT
  cg_maxit::INT
  arcsearch_max_steps::INT
  sif_file_device::INT
  weight::T
  infinity::T
  stop_d::T
  identical_bounds_tol::T
  stop_cg_relative::T
  stop_cg_absolute::T
  alpha_max::T
  alpha_initial::T
  alpha_reduction::T
  arcsearch_acceptance_tol::T
  stabilisation_weight::T
  cpu_time_limit::T
  direct_subproblem_solve::Bool
  exact_arc_search::Bool
  advance::Bool
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T,INT}
  convert_control::convert_control_type{INT}
end

export blls_time_type

struct blls_time_type{T}
  total::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T
end

export blls_inform_type

struct blls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  factorization_status::INT
  iter::INT
  cg_iter::INT
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::blls_time_type{T}
  sbls_inform::sbls_inform_type{T,INT}
  convert_inform::convert_inform_type{T,INT}
end

export blls_initialize

function blls_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{blls_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function blls_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{blls_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function blls_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{blls_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function blls_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{blls_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function blls_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{blls_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function blls_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{blls_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export blls_read_specfile

function blls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.blls_read_specfile(control::Ptr{blls_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.blls_read_specfile(control::Ptr{blls_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.blls_read_specfile(control::Ptr{blls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.blls_read_specfile(control::Ptr{blls_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.blls_read_specfile(control::Ptr{blls_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.blls_read_specfile(control::Ptr{blls_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export blls_import

function blls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single.blls_import(control::Ptr{blls_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       o::Int32, Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                       Ao_row::Ptr{Int32}, Ao_col::Ptr{Int32},
                                       Ao_ptr_ne::Int32, Ao_ptr::Ptr{Int32})::Cvoid
end

function blls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single_64.blls_import(control::Ptr{blls_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, o::Int64, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int64, Ao_row::Ptr{Int64},
                                          Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                          Ao_ptr::Ptr{Int64})::Cvoid
end

function blls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double.blls_import(control::Ptr{blls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       o::Int32, Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                       Ao_row::Ptr{Int32}, Ao_col::Ptr{Int32},
                                       Ao_ptr_ne::Int32, Ao_ptr::Ptr{Int32})::Cvoid
end

function blls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double_64.blls_import(control::Ptr{blls_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, o::Int64, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int64, Ao_row::Ptr{Int64},
                                          Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                          Ao_ptr::Ptr{Int64})::Cvoid
end

function blls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple.blls_import(control::Ptr{blls_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, o::Int32, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int32, Ao_row::Ptr{Int32},
                                          Ao_col::Ptr{Int32}, Ao_ptr_ne::Int32,
                                          Ao_ptr::Ptr{Int32})::Cvoid
end

function blls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, o, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple_64.blls_import(control::Ptr{blls_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, o::Int64, Ao_type::Ptr{Cchar},
                                             Ao_ne::Int64, Ao_row::Ptr{Int64},
                                             Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                             Ao_ptr::Ptr{Int64})::Cvoid
end

export blls_import_without_a

function blls_import_without_a(::Type{Float32}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_single.blls_import_without_a(control::Ptr{blls_control_type{Float32,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, o::Int32)::Cvoid
end

function blls_import_without_a(::Type{Float32}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_single_64.blls_import_without_a(control::Ptr{blls_control_type{Float32,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    o::Int64)::Cvoid
end

function blls_import_without_a(::Type{Float64}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_double.blls_import_without_a(control::Ptr{blls_control_type{Float64,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, o::Int32)::Cvoid
end

function blls_import_without_a(::Type{Float64}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_double_64.blls_import_without_a(control::Ptr{blls_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    o::Int64)::Cvoid
end

function blls_import_without_a(::Type{Float128}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_quadruple.blls_import_without_a(control::Ptr{blls_control_type{Float128,
                                                                                   Int32}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    o::Int32)::Cvoid
end

function blls_import_without_a(::Type{Float128}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_quadruple_64.blls_import_without_a(control::Ptr{blls_control_type{Float128,
                                                                                      Int64}},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       o::Int64)::Cvoid
end

export blls_reset_control

function blls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.blls_reset_control(control::Ptr{blls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function blls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.blls_reset_control(control::Ptr{blls_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function blls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.blls_reset_control(control::Ptr{blls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function blls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.blls_reset_control(control::Ptr{blls_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function blls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.blls_reset_control(control::Ptr{blls_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function blls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.blls_reset_control(control::Ptr{blls_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export blls_solve_given_a

function blls_solve_given_a(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_single.blls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, o::Int32,
                                              Ao_ne::Int32, Ao_val::Ptr{Float32},
                                              b::Ptr{Float32}, x_l::Ptr{Float32},
                                              x_u::Ptr{Float32}, x::Ptr{Float32},
                                              z::Ptr{Float32}, r::Ptr{Float32},
                                              g::Ptr{Float32}, x_stat::Ptr{Int32},
                                              w::Ptr{Float32}, eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_single_64.blls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, o::Int64, Ao_ne::Int64,
                                                 Ao_val::Ptr{Float32}, b::Ptr{Float32},
                                                 x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                 x::Ptr{Float32}, z::Ptr{Float32},
                                                 r::Ptr{Float32}, g::Ptr{Float32},
                                                 x_stat::Ptr{Int64}, w::Ptr{Float32},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_double.blls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, o::Int32,
                                              Ao_ne::Int32, Ao_val::Ptr{Float64},
                                              b::Ptr{Float64}, x_l::Ptr{Float64},
                                              x_u::Ptr{Float64}, x::Ptr{Float64},
                                              z::Ptr{Float64}, r::Ptr{Float64},
                                              g::Ptr{Float64}, x_stat::Ptr{Int32},
                                              w::Ptr{Float64}, eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_double_64.blls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, o::Int64, Ao_ne::Int64,
                                                 Ao_val::Ptr{Float64}, b::Ptr{Float64},
                                                 x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                 x::Ptr{Float64}, z::Ptr{Float64},
                                                 r::Ptr{Float64}, g::Ptr{Float64},
                                                 x_stat::Ptr{Int64}, w::Ptr{Float64},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_quadruple.blls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, o::Int32, Ao_ne::Int32,
                                                 Ao_val::Ptr{Float128}, b::Ptr{Float128},
                                                 x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                                 x::Ptr{Float128}, z::Ptr{Float128},
                                                 r::Ptr{Float128}, g::Ptr{Float128},
                                                 x_stat::Ptr{Int32}, w::Ptr{Float128},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_quadruple_64.blls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64, o::Int64,
                                                    Ao_ne::Int64, Ao_val::Ptr{Float128},
                                                    b::Ptr{Float128}, x_l::Ptr{Float128},
                                                    x_u::Ptr{Float128}, x::Ptr{Float128},
                                                    z::Ptr{Float128}, r::Ptr{Float128},
                                                    g::Ptr{Float128}, x_stat::Ptr{Int64},
                                                    w::Ptr{Float128},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

export blls_solve_reverse_a_prod

function blls_solve_reverse_a_prod(::Type{Float32}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_single.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32},
                                                     eval_status::Ptr{Int32}, n::Int32,
                                                     o::Int32, b::Ptr{Float32},
                                                     x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                     x::Ptr{Float32}, z::Ptr{Float32},
                                                     r::Ptr{Float32}, g::Ptr{Float32},
                                                     x_stat::Ptr{Int32}, v::Ptr{Float32},
                                                     p::Ptr{Float32}, nz_v::Ptr{Int32},
                                                     nz_v_start::Ptr{Int32},
                                                     nz_v_end::Ptr{Int32}, nz_p::Ptr{Int32},
                                                     nz_p_end::Int32,
                                                     w::Ptr{Float32})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float32}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_single_64.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64},
                                                        eval_status::Ptr{Int64}, n::Int64,
                                                        o::Int64, b::Ptr{Float32},
                                                        x_l::Ptr{Float32},
                                                        x_u::Ptr{Float32}, x::Ptr{Float32},
                                                        z::Ptr{Float32}, r::Ptr{Float32},
                                                        g::Ptr{Float32}, x_stat::Ptr{Int64},
                                                        v::Ptr{Float32}, p::Ptr{Float32},
                                                        nz_v::Ptr{Int64},
                                                        nz_v_start::Ptr{Int64},
                                                        nz_v_end::Ptr{Int64},
                                                        nz_p::Ptr{Int64}, nz_p_end::Int64,
                                                        w::Ptr{Float32})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float64}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_double.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32},
                                                     eval_status::Ptr{Int32}, n::Int32,
                                                     o::Int32, b::Ptr{Float64},
                                                     x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                     x::Ptr{Float64}, z::Ptr{Float64},
                                                     r::Ptr{Float64}, g::Ptr{Float64},
                                                     x_stat::Ptr{Int32}, v::Ptr{Float64},
                                                     p::Ptr{Float64}, nz_v::Ptr{Int32},
                                                     nz_v_start::Ptr{Int32},
                                                     nz_v_end::Ptr{Int32}, nz_p::Ptr{Int32},
                                                     nz_p_end::Int32,
                                                     w::Ptr{Float64})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float64}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_double_64.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64},
                                                        eval_status::Ptr{Int64}, n::Int64,
                                                        o::Int64, b::Ptr{Float64},
                                                        x_l::Ptr{Float64},
                                                        x_u::Ptr{Float64}, x::Ptr{Float64},
                                                        z::Ptr{Float64}, r::Ptr{Float64},
                                                        g::Ptr{Float64}, x_stat::Ptr{Int64},
                                                        v::Ptr{Float64}, p::Ptr{Float64},
                                                        nz_v::Ptr{Int64},
                                                        nz_v_start::Ptr{Int64},
                                                        nz_v_end::Ptr{Int64},
                                                        nz_p::Ptr{Int64}, nz_p_end::Int64,
                                                        w::Ptr{Float64})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float128}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_quadruple.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32},
                                                        eval_status::Ptr{Int32}, n::Int32,
                                                        o::Int32, b::Ptr{Float128},
                                                        x_l::Ptr{Float128},
                                                        x_u::Ptr{Float128},
                                                        x::Ptr{Float128}, z::Ptr{Float128},
                                                        r::Ptr{Float128}, g::Ptr{Float128},
                                                        x_stat::Ptr{Int32},
                                                        v::Ptr{Float128}, p::Ptr{Float128},
                                                        nz_v::Ptr{Int32},
                                                        nz_v_start::Ptr{Int32},
                                                        nz_v_end::Ptr{Int32},
                                                        nz_p::Ptr{Int32}, nz_p_end::Int32,
                                                        w::Ptr{Float128})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float128}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_quadruple_64.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int64},
                                                           eval_status::Ptr{Int64},
                                                           n::Int64, o::Int64,
                                                           b::Ptr{Float128},
                                                           x_l::Ptr{Float128},
                                                           x_u::Ptr{Float128},
                                                           x::Ptr{Float128},
                                                           z::Ptr{Float128},
                                                           r::Ptr{Float128},
                                                           g::Ptr{Float128},
                                                           x_stat::Ptr{Int64},
                                                           v::Ptr{Float128},
                                                           p::Ptr{Float128},
                                                           nz_v::Ptr{Int64},
                                                           nz_v_start::Ptr{Int64},
                                                           nz_v_end::Ptr{Int64},
                                                           nz_p::Ptr{Int64},
                                                           nz_p_end::Int64,
                                                           w::Ptr{Float128})::Cvoid
end

export blls_information

function blls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.blls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{blls_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function blls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.blls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{blls_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function blls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.blls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{blls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function blls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.blls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{blls_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function blls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.blls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{blls_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function blls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.blls_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{blls_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export blls_terminate

function blls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{blls_control_type{Float32,Int32}},
                                          inform::Ptr{blls_inform_type{Float32,Int32}})::Cvoid
end

function blls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{blls_control_type{Float32,Int64}},
                                             inform::Ptr{blls_inform_type{Float32,Int64}})::Cvoid
end

function blls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{blls_control_type{Float64,Int32}},
                                          inform::Ptr{blls_inform_type{Float64,Int32}})::Cvoid
end

function blls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{blls_control_type{Float64,Int64}},
                                             inform::Ptr{blls_inform_type{Float64,Int64}})::Cvoid
end

function blls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{blls_control_type{Float128,Int32}},
                                             inform::Ptr{blls_inform_type{Float128,Int32}})::Cvoid
end

function blls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{blls_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{blls_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
