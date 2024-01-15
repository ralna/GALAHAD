export blls_control_type

mutable struct blls_control_type{T}
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

  function blls_control_type{T}() where T
    type = new()
    type.sbls_control = sbls_control_type{T}()
    type.convert_control = convert_control_type()
    return type
  end
end

export blls_time_type

mutable struct blls_time_type{T}
  total::T
  analyse::T
  factorize::T
  solve::T
  clock_total::T
  clock_analyse::T
  clock_factorize::T
  clock_solve::T

  blls_time_type{T}() where T = new()
end

export blls_inform_type

mutable struct blls_inform_type{T}
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

  function blls_inform_type{T}() where T
    type = new()
    type.time = blls_time_type{T}()
    type.sbls_inform = sbls_inform_type{T}()
    type.convert_inform = convert_inform_type{T}()
    return type
  end
end

export blls_initialize_s

function blls_initialize_s(data, control, status)
  @ccall libgalahad_single.blls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{blls_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export blls_initialize

function blls_initialize(data, control, status)
  @ccall libgalahad_double.blls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{blls_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export blls_read_specfile_s

function blls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.blls_read_specfile_s(control::Ref{blls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export blls_read_specfile

function blls_read_specfile(control, specfile)
  @ccall libgalahad_double.blls_read_specfile(control::Ref{blls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export blls_import_s

function blls_import_s(control, data, status, n, o,
                       Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_single.blls_import_s(control::Ref{blls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                         Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                         Ao_ptr_ne::Cint,
                                         Ao_ptr::Ptr{Cint})::Cvoid
end

export blls_import

function blls_import(control, data, status, n, o,
                     Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr)
  @ccall libgalahad_double.blls_import(control::Ref{blls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       o::Cint, Ao_type::Ptr{Cchar}, Ao_ne::Cint,
                                       Ao_row::Ptr{Cint}, Ao_col::Ptr{Cint},
                                       Ao_ptr_ne::Cint,
                                       Ao_ptr::Ptr{Cint})::Cvoid
end

export blls_import_without_a_s

function blls_import_without_a_s(control, data, status, n, o)
  @ccall libgalahad_single.blls_import_without_a_s(control::Ref{blls_control_type{Float32}},
                                                   data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, o::Cint)::Cvoid
end

export blls_import_without_a

function blls_import_without_a(control, data, status, n, o)
  @ccall libgalahad_double.blls_import_without_a(control::Ref{blls_control_type{Float64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, o::Cint)::Cvoid
end

export blls_reset_control_s

function blls_reset_control_s(control, data, status)
  @ccall libgalahad_single.blls_reset_control_s(control::Ref{blls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export blls_reset_control

function blls_reset_control(control, data, status)
  @ccall libgalahad_double.blls_reset_control(control::Ref{blls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export blls_solve_given_a_s

function blls_solve_given_a_s(data, userdata, status, n, o, Ao_ne, Ao_val, b, x_l, x_u, x, z, r,
                            g, x_stat, w, eval_prec)
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

export blls_solve_given_a

function blls_solve_given_a(data, userdata, status, n, o, Ao_ne, Ao_val, b, x_l, x_u, x, z, r,
                          g, x_stat, w, eval_prec)
  @ccall libgalahad_double.blls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint, o::Cint,
                                              Ao_ne::Cint, Ao_val::Ptr{Float64},
                                              b::Ptr{Float64}, x_l::Ptr{Float64},
                                              x_u::Ptr{Float64}, x::Ptr{Float64},
                                              z::Ptr{Float64}, r::Ptr{Float64},
                                              g::Ptr{Float64}, x_stat::Ptr{Cint},
                                              w::Ptr{Float64},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

export blls_solve_reverse_a_prod_s

function blls_solve_reverse_a_prod_s(data, status, eval_status, n, o, b, x_l, x_u, x, z, r, g,
                                   x_stat, v, p, nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end,
                                   w)
  @ccall libgalahad_single.blls_solve_reverse_a_prod_s(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint},
                                                       eval_status::Ptr{Cint}, n::Cint,
                                                       o::Cint, b::Ptr{Float32},
                                                       x_l::Ptr{Float32},
                                                       x_u::Ptr{Float32}, x::Ptr{Float32},
                                                       z::Ptr{Float32}, r::Ptr{Float32},
                                                       g::Ptr{Float32}, x_stat::Ptr{Cint},
                                                       v::Ptr{Float32}, p::Ptr{Float32},
                                                       nz_v::Ptr{Cint},
                                                       nz_v_start::Ptr{Cint},
                                                       nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                       nz_p_end::Cint,
                                                       w::Ptr{Float32})::Cvoid
end

export blls_solve_reverse_a_prod

function blls_solve_reverse_a_prod(data, status, eval_status, n, o, b, x_l, x_u, x, z, r, g,
                                 x_stat, v, p, nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end,
                                 w)
  @ccall libgalahad_double.blls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Cint},
                                                     eval_status::Ptr{Cint}, n::Cint,
                                                     o::Cint, b::Ptr{Float64},
                                                     x_l::Ptr{Float64},
                                                     x_u::Ptr{Float64}, x::Ptr{Float64},
                                                     z::Ptr{Float64}, r::Ptr{Float64},
                                                     g::Ptr{Float64}, x_stat::Ptr{Cint},
                                                     v::Ptr{Float64}, p::Ptr{Float64},
                                                     nz_v::Ptr{Cint},
                                                     nz_v_start::Ptr{Cint},
                                                     nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                     nz_p_end::Cint,
                                                     w::Ptr{Float64})::Cvoid
end

export blls_information_s

function blls_information_s(data, inform, status)
  @ccall libgalahad_single.blls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{blls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export blls_information

function blls_information(data, inform, status)
  @ccall libgalahad_double.blls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{blls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export blls_terminate_s

function blls_terminate_s(data, control, inform)
  @ccall libgalahad_single.blls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{blls_control_type{Float32}},
                                            inform::Ref{blls_inform_type{Float32}})::Cvoid
end

export blls_terminate

function blls_terminate(data, control, inform)
  @ccall libgalahad_double.blls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{blls_control_type{Float64}},
                                          inform::Ref{blls_inform_type{Float64}})::Cvoid
end
