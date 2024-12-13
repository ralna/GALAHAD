export blls_control_type

struct blls_control_type{T}
  f_indexing::Bool
  error::Cint
  out::Cint
  print_level::Cint
  start_print::Cint
  stop_print::Cint
  print_gap::Cint
  maxit::Cint
  cold_start::Cint
  preconditioner::Cint
  ratio_cg_vs_sd::Cint
  change_max::Cint
  cg_maxit::Cint
  arcsearch_max_steps::Cint
  sif_file_device::Cint
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
  sbls_control::sbls_control_type{T}
  convert_control::convert_control_type
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

struct blls_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorization_status::Cint
  iter::Cint
  cg_iter::Cint
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::blls_time_type{T}
  sbls_inform::sbls_inform_type{T}
  convert_inform::convert_inform_type{T}
end

export blls_initialize

function blls_initialize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.blls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{blls_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

function blls_initialize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{blls_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

function blls_initialize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.blls_initialize_q(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{blls_control_type{Float128}},
                                                status::Ptr{Cint})::Cvoid
end

export blls_read_specfile

function blls_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.blls_read_specfile_s(control::Ptr{blls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.blls_read_specfile(control::Ptr{blls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

function blls_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.blls_read_specfile_q(control::Ptr{blls_control_type{Float128}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export blls_import

function blls_import(::Type{Float32}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                     Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single.blls_import_s(control::Ptr{blls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                         Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                         Ao_ptr_ne::Cint, Ao_ptr::Ptr{Cint})::Cvoid
end

function blls_import(::Type{Float64}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                     Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double.blls_import(control::Ptr{blls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                       Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                       Ao_ptr_ne::Cint, Ao_ptr::Ptr{Cint})::Cvoid
end

function blls_import(::Type{Float128}, control, data, status, n, o, Ao_type, Ao_ne, Ao_row,
                     Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_quadruple.blls_import_q(control::Ptr{blls_control_type{Float128}},
                                            data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, o::Cint, Ao_type::Ptr{Cchar},
                                            Ao_ne::Cint, Ao_row::Ptr{Cint},
                                            Ao_col::Ptr{Cint}, Ao_ptr_ne::Cint,
                                            Ao_ptr::Ptr{Cint})::Cvoid
end

export blls_import_without_a

function blls_import_without_a(::Type{Float32}, control, data, status, n, o)
  @ccall libgalahad_single.blls_import_without_a_s(control::Ptr{blls_control_type{Float32}},
                                                   data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, o::Cint)::Cvoid
end

function blls_import_without_a(::Type{Float64}, control, data, status, n, o)
  @ccall libgalahad_double.blls_import_without_a(control::Ptr{blls_control_type{Float64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, o::Cint)::Cvoid
end

function blls_import_without_a(::Type{Float128}, control, data, status, n, o)
  @ccall libgalahad_quadruple.blls_import_without_a_q(control::Ptr{blls_control_type{Float128}},
                                                      data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      o::Cint)::Cvoid
end

export blls_reset_control

function blls_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.blls_reset_control_s(control::Ptr{blls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function blls_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.blls_reset_control(control::Ptr{blls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

function blls_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.blls_reset_control_q(control::Ptr{blls_control_type{Float128}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Cint})::Cvoid
end

export blls_solve_given_a

function blls_solve_given_a(::Type{Float32}, data, userdata, status, n, o, Ao_ne, Ao_val, b,
                            x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_single.blls_solve_given_a_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint, o::Cint,
                                                Ao_ne::Cint, Ao_val::Ptr{Float32},
                                                b::Ptr{Float32}, x_l::Ptr{Float32},
                                                x_u::Ptr{Float32}, x::Ptr{Float32},
                                                z::Ptr{Float32}, r::Ptr{Float32},
                                                g::Ptr{Float32}, x_stat::Ptr{Cint},
                                                w::Ptr{Float32},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float64}, data, userdata, status, n, o, Ao_ne, Ao_val, b,
                            x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_double.blls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint, o::Cint,
                                              Ao_ne::Cint, Ao_val::Ptr{Float64},
                                              b::Ptr{Float64}, x_l::Ptr{Float64},
                                              x_u::Ptr{Float64}, x::Ptr{Float64},
                                              z::Ptr{Float64}, r::Ptr{Float64},
                                              g::Ptr{Float64}, x_stat::Ptr{Cint},
                                              w::Ptr{Float64}, eval_prec::Ptr{Cvoid})::Cvoid
end

function blls_solve_given_a(::Type{Float128}, data, userdata, status, n, o, Ao_ne, Ao_val,
                            b, x_l, x_u, x, z, r, g, x_stat, w, eval_prec)
  @ccall libgalahad_quadruple.blls_solve_given_a_q(data::Ptr{Ptr{Cvoid}},
                                                   userdata::Ptr{Cvoid}, status::Ptr{Cint},
                                                   n::Cint, o::Cint, Ao_ne::Cint,
                                                   Ao_val::Ptr{Float128}, b::Ptr{Float128},
                                                   x_l::Ptr{Float128}, x_u::Ptr{Float128},
                                                   x::Ptr{Float128}, z::Ptr{Float128},
                                                   r::Ptr{Float128}, g::Ptr{Float128},
                                                   x_stat::Ptr{Cint}, w::Ptr{Float128},
                                                   eval_prec::Ptr{Cvoid})::Cvoid
end

export blls_solve_reverse_a_prod

function blls_solve_reverse_a_prod(::Type{Float32}, data, status, eval_status, n, o, b, x_l,
                                   x_u, x, z, r, g, x_stat, v, p, nz_v, nz_v_start,
                                   nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_single.blls_solve_reverse_a_prod_s(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint},
                                                       eval_status::Ptr{Cint}, n::Cint,
                                                       o::Cint, b::Ptr{Float32},
                                                       x_l::Ptr{Float32}, x_u::Ptr{Float32},
                                                       x::Ptr{Float32}, z::Ptr{Float32},
                                                       r::Ptr{Float32}, g::Ptr{Float32},
                                                       x_stat::Ptr{Cint}, v::Ptr{Float32},
                                                       p::Ptr{Float32}, nz_v::Ptr{Cint},
                                                       nz_v_start::Ptr{Cint},
                                                       nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                       nz_p_end::Cint,
                                                       w::Ptr{Float32})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float64}, data, status, eval_status, n, o, b, x_l,
                                   x_u, x, z, r, g, x_stat, v, p, nz_v, nz_v_start,
                                   nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_double.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Cint},
                                                     eval_status::Ptr{Cint}, n::Cint,
                                                     o::Cint, b::Ptr{Float64},
                                                     x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                     x::Ptr{Float64}, z::Ptr{Float64},
                                                     r::Ptr{Float64}, g::Ptr{Float64},
                                                     x_stat::Ptr{Cint}, v::Ptr{Float64},
                                                     p::Ptr{Float64}, nz_v::Ptr{Cint},
                                                     nz_v_start::Ptr{Cint},
                                                     nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                     nz_p_end::Cint, w::Ptr{Float64})::Cvoid
end

function blls_solve_reverse_a_prod(::Type{Float128}, data, status, eval_status, n, o, b,
                                   x_l, x_u, x, z, r, g, x_stat, v, p, nz_v, nz_v_start,
                                   nz_v_end, nz_p, nz_p_end, w)
  @ccall libgalahad_quadruple.blls_solve_reverse_a_prod_q(data::Ptr{Ptr{Cvoid}},
                                                          status::Ptr{Cint},
                                                          eval_status::Ptr{Cint}, n::Cint,
                                                          o::Cint, b::Ptr{Float128},
                                                          x_l::Ptr{Float128},
                                                          x_u::Ptr{Float128},
                                                          x::Ptr{Float128},
                                                          z::Ptr{Float128},
                                                          r::Ptr{Float128},
                                                          g::Ptr{Float128},
                                                          x_stat::Ptr{Cint},
                                                          v::Ptr{Float128},
                                                          p::Ptr{Float128}, nz_v::Ptr{Cint},
                                                          nz_v_start::Ptr{Cint},
                                                          nz_v_end::Ptr{Cint},
                                                          nz_p::Ptr{Cint}, nz_p_end::Cint,
                                                          w::Ptr{Float128})::Cvoid
end

export blls_information

function blls_information(::Type{Float32}, data, inform, status)
  @ccall libgalahad_single.blls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{blls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

function blls_information(::Type{Float64}, data, inform, status)
  @ccall libgalahad_double.blls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{blls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

function blls_information(::Type{Float128}, data, inform, status)
  @ccall libgalahad_quadruple.blls_information_q(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ptr{blls_inform_type{Float128}},
                                                 status::Ptr{Cint})::Cvoid
end

export blls_terminate

function blls_terminate(::Type{Float32}, data, control, inform)
  @ccall libgalahad_single.blls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{blls_control_type{Float32}},
                                            inform::Ptr{blls_inform_type{Float32}})::Cvoid
end

function blls_terminate(::Type{Float64}, data, control, inform)
  @ccall libgalahad_double.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{blls_control_type{Float64}},
                                          inform::Ptr{blls_inform_type{Float64}})::Cvoid
end

function blls_terminate(::Type{Float128}, data, control, inform)
  @ccall libgalahad_quadruple.blls_terminate_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{blls_control_type{Float128}},
                                               inform::Ptr{blls_inform_type{Float128}})::Cvoid
end
