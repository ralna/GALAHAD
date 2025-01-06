export slls_control_type

struct slls_control_type{T,INT}
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
  stop_d::T
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
  space_critical::Bool
  deallocate_error_fatal::Bool
  generate_sif_file::Bool
  sif_file_name::NTuple{31,Cchar}
  prefix::NTuple{31,Cchar}
  sbls_control::sbls_control_type{T,INT}
  convert_control::convert_control_type{INT}
end

export slls_time_type

struct slls_time_type
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32
end

export slls_inform_type

struct slls_inform_type{T,INT}
  status::INT
  alloc_status::INT
  factorization_status::INT
  iter::INT
  cg_iter::INT
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::slls_time_type
  sbls_inform::sbls_inform_type{T,INT}
  convert_inform::convert_inform_type{T,INT}
end

export slls_initialize

function slls_initialize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{slls_control_type{Float32,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function slls_initialize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slls_control_type{Float32,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function slls_initialize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{slls_control_type{Float64,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function slls_initialize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slls_control_type{Float64,Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function slls_initialize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slls_control_type{Float128,
                                                                             Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function slls_initialize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{slls_control_type{Float128,
                                                                                Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export slls_read_specfile

function slls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.slls_read_specfile(control::Ptr{slls_control_type{Float32,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function slls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.slls_read_specfile(control::Ptr{slls_control_type{Float32,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function slls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.slls_read_specfile(control::Ptr{slls_control_type{Float64,Int32}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function slls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.slls_read_specfile(control::Ptr{slls_control_type{Float64,
                                                                                Int64}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function slls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.slls_read_specfile(control::Ptr{slls_control_type{Float128,
                                                                                Int32}},
                                                 specfile::Ptr{Cchar})::Cvoid
end

function slls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.slls_read_specfile(control::Ptr{slls_control_type{Float128,
                                                                                   Int64}},
                                                    specfile::Ptr{Cchar})::Cvoid
end

export slls_import

function slls_import(::Type{Float32}, ::Type{Int32}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single.slls_import(control::Ptr{slls_control_type{Float32,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                       Ao_row::Ptr{Int32}, Ao_col::Ptr{Int32},
                                       Ao_ptr_ne::Int32, Ao_ptr::Ptr{Int32})::Cvoid
end

function slls_import(::Type{Float32}, ::Type{Int64}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single_64.slls_import(control::Ptr{slls_control_type{Float32,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int64, Ao_row::Ptr{Int64},
                                          Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                          Ao_ptr::Ptr{Int64})::Cvoid
end

function slls_import(::Type{Float64}, ::Type{Int32}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double.slls_import(control::Ptr{slls_control_type{Float64,Int32}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32}, n::Int32,
                                       m::Int32, Ao_type::Ptr{Cchar}, Ao_ne::Int32,
                                       Ao_row::Ptr{Int32}, Ao_col::Ptr{Int32},
                                       Ao_ptr_ne::Int32, Ao_ptr::Ptr{Int32})::Cvoid
end

function slls_import(::Type{Float64}, ::Type{Int64}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double_64.slls_import(control::Ptr{slls_control_type{Float64,Int64}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                          n::Int64, m::Int64, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int64, Ao_row::Ptr{Int64},
                                          Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                          Ao_ptr::Ptr{Int64})::Cvoid
end

function slls_import(::Type{Float128}, ::Type{Int32}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple.slls_import(control::Ptr{slls_control_type{Float128,Int32}},
                                          data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                          n::Int32, m::Int32, Ao_type::Ptr{Cchar},
                                          Ao_ne::Int32, Ao_row::Ptr{Int32},
                                          Ao_col::Ptr{Int32}, Ao_ptr_ne::Int32,
                                          Ao_ptr::Ptr{Int32})::Cvoid
end

function slls_import(::Type{Float128}, ::Type{Int64}, control, data, status, n, m, Ao_type,
                     Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple_64.slls_import(control::Ptr{slls_control_type{Float128,Int64}},
                                             data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64},
                                             n::Int64, m::Int64, Ao_type::Ptr{Cchar},
                                             Ao_ne::Int64, Ao_row::Ptr{Int64},
                                             Ao_col::Ptr{Int64}, Ao_ptr_ne::Int64,
                                             Ao_ptr::Ptr{Int64})::Cvoid
end

export slls_import_without_a

function slls_import_without_a(::Type{Float32}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_single.slls_import_without_a(control::Ptr{slls_control_type{Float32,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, o::Int32)::Cvoid
end

function slls_import_without_a(::Type{Float32}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_single_64.slls_import_without_a(control::Ptr{slls_control_type{Float32,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    o::Int64)::Cvoid
end

function slls_import_without_a(::Type{Float64}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_double.slls_import_without_a(control::Ptr{slls_control_type{Float64,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                                 n::Int32, o::Int32)::Cvoid
end

function slls_import_without_a(::Type{Float64}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_double_64.slls_import_without_a(control::Ptr{slls_control_type{Float64,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64}, n::Int64,
                                                    o::Int64)::Cvoid
end

function slls_import_without_a(::Type{Float128}, ::Type{Int32}, control, data, status, n, o)
  @ccall libgalahad_quadruple.slls_import_without_a(control::Ptr{slls_control_type{Float128,
                                                                                   Int32}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int32}, n::Int32,
                                                    o::Int32)::Cvoid
end

function slls_import_without_a(::Type{Float128}, ::Type{Int64}, control, data, status, n, o)
  @ccall libgalahad_quadruple_64.slls_import_without_a(control::Ptr{slls_control_type{Float128,
                                                                                      Int64}},
                                                       data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Int64}, n::Int64,
                                                       o::Int64)::Cvoid
end

export slls_reset_control

function slls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.slls_reset_control(control::Ptr{slls_control_type{Float32,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function slls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.slls_reset_control(control::Ptr{slls_control_type{Float32,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function slls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.slls_reset_control(control::Ptr{slls_control_type{Float64,Int32}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Int32})::Cvoid
end

function slls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.slls_reset_control(control::Ptr{slls_control_type{Float64,
                                                                                Int64}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int64})::Cvoid
end

function slls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.slls_reset_control(control::Ptr{slls_control_type{Float128,
                                                                                Int32}},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Int32})::Cvoid
end

function slls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.slls_reset_control(control::Ptr{slls_control_type{Float128,
                                                                                   Int64}},
                                                    data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Int64})::Cvoid
end

export slls_solve_given_a

function slls_solve_given_a(::Type{Float32}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_single.slls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, o::Int32,
                                              Ao_ne::Int32, Ao_val::Ptr{Float32},
                                              b::Ptr{Float32}, x::Ptr{Float32},
                                              z::Ptr{Float32}, r::Ptr{Float32},
                                              g::Ptr{Float32}, x_stat::Ptr{Int32},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function slls_solve_given_a(::Type{Float32}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_single_64.slls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, o::Int64, Ao_ne::Int64,
                                                 Ao_val::Ptr{Float32}, b::Ptr{Float32},
                                                 x::Ptr{Float32}, z::Ptr{Float32},
                                                 r::Ptr{Float32}, g::Ptr{Float32},
                                                 x_stat::Ptr{Int64},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function slls_solve_given_a(::Type{Float64}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_double.slls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, n::Int32, o::Int32,
                                              Ao_ne::Int32, Ao_val::Ptr{Float64},
                                              b::Ptr{Float64}, x::Ptr{Float64},
                                              z::Ptr{Float64}, r::Ptr{Float64},
                                              g::Ptr{Float64}, x_stat::Ptr{Int32},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

function slls_solve_given_a(::Type{Float64}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_double_64.slls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int64},
                                                 n::Int64, o::Int64, Ao_ne::Int64,
                                                 Ao_val::Ptr{Float64}, b::Ptr{Float64},
                                                 x::Ptr{Float64}, z::Ptr{Float64},
                                                 r::Ptr{Float64}, g::Ptr{Float64},
                                                 x_stat::Ptr{Int64},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function slls_solve_given_a(::Type{Float128}, ::Type{Int32}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_quadruple.slls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                 userdata::Ptr{Cvoid}, status::Ptr{Int32},
                                                 n::Int32, o::Int32, Ao_ne::Int32,
                                                 Ao_val::Ptr{Float128}, b::Ptr{Float128},
                                                 x::Ptr{Float128}, z::Ptr{Float128},
                                                 r::Ptr{Float128}, g::Ptr{Float128},
                                                 x_stat::Ptr{Int32},
                                                 eval_prec::Ptr{Cvoid})::Cvoid
end

function slls_solve_given_a(::Type{Float128}, ::Type{Int64}, data, userdata, status, n, o,
                            Ao_ne, Ao_val, b, x, z, r, g, x_stat, eval_prec)
  @ccall libgalahad_quadruple_64.slls_solve_given_a(data::Ptr{Ptr{Cvoid}},
                                                    userdata::Ptr{Cvoid},
                                                    status::Ptr{Int64}, n::Int64, o::Int64,
                                                    Ao_ne::Int64, Ao_val::Ptr{Float128},
                                                    b::Ptr{Float128}, x::Ptr{Float128},
                                                    z::Ptr{Float128}, r::Ptr{Float128},
                                                    g::Ptr{Float128}, x_stat::Ptr{Int64},
                                                    eval_prec::Ptr{Cvoid})::Cvoid
end

export slls_solve_reverse_a_prod

function slls_solve_reverse_a_prod(::Type{Float32}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_single.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32},
                                                     eval_status::Ptr{Int32}, n::Int32,
                                                     o::Int32, b::Ptr{Float32},
                                                     x::Ptr{Float32}, z::Ptr{Float32},
                                                     r::Ptr{Float32}, g::Ptr{Float32},
                                                     x_stat::Ptr{Int32}, v::Ptr{Float32},
                                                     p::Ptr{Float32}, nz_v::Ptr{Int32},
                                                     nz_v_start::Ptr{Int32},
                                                     nz_v_end::Ptr{Int32}, nz_p::Ptr{Int32},
                                                     nz_p_end::Int32)::Cvoid
end

function slls_solve_reverse_a_prod(::Type{Float32}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_single_64.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64},
                                                        eval_status::Ptr{Int64}, n::Int64,
                                                        o::Int64, b::Ptr{Float32},
                                                        x::Ptr{Float32}, z::Ptr{Float32},
                                                        r::Ptr{Float32}, g::Ptr{Float32},
                                                        x_stat::Ptr{Int64}, v::Ptr{Float32},
                                                        p::Ptr{Float32}, nz_v::Ptr{Int64},
                                                        nz_v_start::Ptr{Int64},
                                                        nz_v_end::Ptr{Int64},
                                                        nz_p::Ptr{Int64},
                                                        nz_p_end::Int64)::Cvoid
end

function slls_solve_reverse_a_prod(::Type{Float64}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_double.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Int32},
                                                     eval_status::Ptr{Int32}, n::Int32,
                                                     o::Int32, b::Ptr{Float64},
                                                     x::Ptr{Float64}, z::Ptr{Float64},
                                                     r::Ptr{Float64}, g::Ptr{Float64},
                                                     x_stat::Ptr{Int32}, v::Ptr{Float64},
                                                     p::Ptr{Float64}, nz_v::Ptr{Int32},
                                                     nz_v_start::Ptr{Int32},
                                                     nz_v_end::Ptr{Int32}, nz_p::Ptr{Int32},
                                                     nz_p_end::Int32)::Cvoid
end

function slls_solve_reverse_a_prod(::Type{Float64}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_double_64.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int64},
                                                        eval_status::Ptr{Int64}, n::Int64,
                                                        o::Int64, b::Ptr{Float64},
                                                        x::Ptr{Float64}, z::Ptr{Float64},
                                                        r::Ptr{Float64}, g::Ptr{Float64},
                                                        x_stat::Ptr{Int64}, v::Ptr{Float64},
                                                        p::Ptr{Float64}, nz_v::Ptr{Int64},
                                                        nz_v_start::Ptr{Int64},
                                                        nz_v_end::Ptr{Int64},
                                                        nz_p::Ptr{Int64},
                                                        nz_p_end::Int64)::Cvoid
end

function slls_solve_reverse_a_prod(::Type{Float128}, ::Type{Int32}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_quadruple.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Int32},
                                                        eval_status::Ptr{Int32}, n::Int32,
                                                        o::Int32, b::Ptr{Float128},
                                                        x::Ptr{Float128}, z::Ptr{Float128},
                                                        r::Ptr{Float128}, g::Ptr{Float128},
                                                        x_stat::Ptr{Int32},
                                                        v::Ptr{Float128}, p::Ptr{Float128},
                                                        nz_v::Ptr{Int32},
                                                        nz_v_start::Ptr{Int32},
                                                        nz_v_end::Ptr{Int32},
                                                        nz_p::Ptr{Int32},
                                                        nz_p_end::Int32)::Cvoid
end

function slls_solve_reverse_a_prod(::Type{Float128}, ::Type{Int64}, data, status,
                                   eval_status, n, o, b, x, z, r, g, x_stat, v, p, nz_v,
                                   nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_quadruple_64.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                           status::Ptr{Int64},
                                                           eval_status::Ptr{Int64},
                                                           n::Int64, o::Int64,
                                                           b::Ptr{Float128},
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
                                                           nz_p_end::Int64)::Cvoid
end

export slls_information

function slls_information(::Type{Float32}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_single.slls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{slls_inform_type{Float32,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function slls_information(::Type{Float32}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_single_64.slls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{slls_inform_type{Float32,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function slls_information(::Type{Float64}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_double.slls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{slls_inform_type{Float64,Int32}},
                                            status::Ptr{Int32})::Cvoid
end

function slls_information(::Type{Float64}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_double_64.slls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{slls_inform_type{Float64,Int64}},
                                               status::Ptr{Int64})::Cvoid
end

function slls_information(::Type{Float128}, ::Type{Int32}, data, inform, status)
  @ccall libgalahad_quadruple.slls_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{slls_inform_type{Float128,Int32}},
                                               status::Ptr{Int32})::Cvoid
end

function slls_information(::Type{Float128}, ::Type{Int64}, data, inform, status)
  @ccall libgalahad_quadruple_64.slls_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ptr{slls_inform_type{Float128,
                                                                               Int64}},
                                                  status::Ptr{Int64})::Cvoid
end

export slls_terminate

function slls_terminate(::Type{Float32}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_single.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{slls_control_type{Float32,Int32}},
                                          inform::Ptr{slls_inform_type{Float32,Int32}})::Cvoid
end

function slls_terminate(::Type{Float32}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_single_64.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{slls_control_type{Float32,Int64}},
                                             inform::Ptr{slls_inform_type{Float32,Int64}})::Cvoid
end

function slls_terminate(::Type{Float64}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_double.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{slls_control_type{Float64,Int32}},
                                          inform::Ptr{slls_inform_type{Float64,Int32}})::Cvoid
end

function slls_terminate(::Type{Float64}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_double_64.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{slls_control_type{Float64,Int64}},
                                             inform::Ptr{slls_inform_type{Float64,Int64}})::Cvoid
end

function slls_terminate(::Type{Float128}, ::Type{Int32}, data, control, inform)
  @ccall libgalahad_quadruple.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{slls_control_type{Float128,Int32}},
                                             inform::Ptr{slls_inform_type{Float128,Int32}})::Cvoid
end

function slls_terminate(::Type{Float128}, ::Type{Int64}, data, control, inform)
  @ccall libgalahad_quadruple_64.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{slls_control_type{Float128,
                                                                               Int64}},
                                                inform::Ptr{slls_inform_type{Float128,
                                                                             Int64}})::Cvoid
end
