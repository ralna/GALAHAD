export slls_control_type

mutable struct slls_control_type{T}
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
  sbls_control::sbls_control_type{T}
  convert_control::convert_control_type

  function slls_control_type{T}() where T
    type = new()
    type.sbls_control = sbls_control_type{T}()
    type.convert_control = convert_control_type()
    return type
  end
end

export slls_time_type

mutable struct slls_time_type
  total::Float32
  analyse::Float32
  factorize::Float32
  solve::Float32

  slls_time_type() = new()
end

export slls_inform_type

mutable struct slls_inform_type{T}
  status::Cint
  alloc_status::Cint
  factorization_status::Cint
  iter::Cint
  cg_iter::Cint
  obj::T
  norm_pg::T
  bad_alloc::NTuple{81,Cchar}
  time::slls_time_type
  sbls_inform::sbls_inform_type{T}
  convert_inform::convert_inform_type{T}

  function slls_inform_type{T}() where T
    type = new()
    type.time = slls_time_type()
    type.sbls_inform = sbls_inform_type{T}()
    type.convert_inform = convert_inform_type{T}()
    return type
  end
end

export slls_initialize_s

function slls_initialize_s(data, control, status)
  @ccall libgalahad_single.slls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{slls_control_type{Float32}},
                                             status::Ptr{Cint})::Cvoid
end

export slls_initialize

function slls_initialize(data, control, status)
  @ccall libgalahad_double.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{slls_control_type{Float64}},
                                           status::Ptr{Cint})::Cvoid
end

export slls_read_specfile_s

function slls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.slls_read_specfile_s(control::Ref{slls_control_type{Float32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

export slls_read_specfile

function slls_read_specfile(control, specfile)
  @ccall libgalahad_double.slls_read_specfile(control::Ref{slls_control_type{Float64}},
                                              specfile::Ptr{Cchar})::Cvoid
end

export slls_import_s

function slls_import_s(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_single.slls_import_s(control::Ref{slls_control_type{Float32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

export slls_import

function slls_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
  @ccall libgalahad_double.slls_import(control::Ref{slls_control_type{Float64}},
                                       data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                       m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                       A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                       A_ptr::Ptr{Cint})::Cvoid
end

export slls_import_without_a_s

function slls_import_without_a_s(control, data, status, n, m)
  @ccall libgalahad_single.slls_import_without_a_s(control::Ref{slls_control_type{Float32}},
                                                   data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, m::Cint)::Cvoid
end

export slls_import_without_a

function slls_import_without_a(control, data, status, n, m)
  @ccall libgalahad_double.slls_import_without_a(control::Ref{slls_control_type{Float64}},
                                                 data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                 n::Cint, m::Cint)::Cvoid
end

export slls_reset_control_s

function slls_reset_control_s(control, data, status)
  @ccall libgalahad_single.slls_reset_control_s(control::Ref{slls_control_type{Float32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

export slls_reset_control

function slls_reset_control(control, data, status)
  @ccall libgalahad_double.slls_reset_control(control::Ref{slls_control_type{Float64}},
                                              data::Ptr{Ptr{Cvoid}},
                                              status::Ptr{Cint})::Cvoid
end

export slls_solve_given_a_s

function slls_solve_given_a_s(data, userdata, status, n, m, A_ne, A_val, b, x, z, c, g,
                            x_stat, eval_prec)
  @ccall libgalahad_single.slls_solve_given_a_s(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                                status::Ptr{Cint}, n::Cint, m::Cint,
                                                A_ne::Cint, A_val::Ptr{Float32},
                                                b::Ptr{Float32}, x::Ptr{Float32},
                                                z::Ptr{Float32}, c::Ptr{Float32},
                                                g::Ptr{Float32}, x_stat::Ptr{Cint},
                                                eval_prec::Ptr{Cvoid})::Cvoid
end

export slls_solve_given_a

function slls_solve_given_a(data, userdata, status, n, m, A_ne, A_val, b, x, z, c, g,
                          x_stat, eval_prec)
  @ccall libgalahad_double.slls_solve_given_a(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, n::Cint, m::Cint,
                                              A_ne::Cint, A_val::Ptr{Float64},
                                              b::Ptr{Float64}, x::Ptr{Float64},
                                              z::Ptr{Float64}, c::Ptr{Float64},
                                              g::Ptr{Float64}, x_stat::Ptr{Cint},
                                              eval_prec::Ptr{Cvoid})::Cvoid
end

export slls_solve_reverse_a_prod_s

function slls_solve_reverse_a_prod_s(data, status, eval_status, n, m, b, x, z, c, g, x_stat,
                                   v, p, nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_single.slls_solve_reverse_a_prod_s(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint},
                                                       eval_status::Ptr{Cint}, n::Cint,
                                                       m::Cint, b::Ptr{Float32},
                                                       x::Ptr{Float32}, z::Ptr{Float32},
                                                       c::Ptr{Float32}, g::Ptr{Float32},
                                                       x_stat::Ptr{Cint}, v::Ptr{Float32},
                                                       p::Ptr{Float32}, nz_v::Ptr{Cint},
                                                       nz_v_start::Ptr{Cint},
                                                       nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                       nz_p_end::Cint)::Cvoid
end

export slls_solve_reverse_a_prod

function slls_solve_reverse_a_prod(data, status, eval_status, n, m, b, x, z, c, g, x_stat,
                                 v, p, nz_v, nz_v_start, nz_v_end, nz_p, nz_p_end)
  @ccall libgalahad_double.slls_solve_reverse_a_prod(data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Cint},
                                                     eval_status::Ptr{Cint}, n::Cint,
                                                     m::Cint, b::Ptr{Float64},
                                                     x::Ptr{Float64}, z::Ptr{Float64},
                                                     c::Ptr{Float64}, g::Ptr{Float64},
                                                     x_stat::Ptr{Cint}, v::Ptr{Float64},
                                                     p::Ptr{Float64}, nz_v::Ptr{Cint},
                                                     nz_v_start::Ptr{Cint},
                                                     nz_v_end::Ptr{Cint}, nz_p::Ptr{Cint},
                                                     nz_p_end::Cint)::Cvoid
end

export slls_information_s

function slls_information_s(data, inform, status)
  @ccall libgalahad_single.slls_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{slls_inform_type{Float32}},
                                              status::Ptr{Cint})::Cvoid
end

export slls_information

function slls_information(data, inform, status)
  @ccall libgalahad_double.slls_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ref{slls_inform_type{Float64}},
                                            status::Ptr{Cint})::Cvoid
end

export slls_terminate_s

function slls_terminate_s(data, control, inform)
  @ccall libgalahad_single.slls_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{slls_control_type{Float32}},
                                            inform::Ref{slls_inform_type{Float32}})::Cvoid
end

export slls_terminate

function slls_terminate(data, control, inform)
  @ccall libgalahad_double.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{slls_control_type{Float64}},
                                          inform::Ref{slls_inform_type{Float64}})::Cvoid
end
