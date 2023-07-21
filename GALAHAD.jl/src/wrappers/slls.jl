mutable struct slls_control_type
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
    weight::Float64
    stop_d::Float64
    stop_cg_relative::Float64
    stop_cg_absolute::Float64
    alpha_max::Float64
    alpha_initial::Float64
    alpha_reduction::Float64
    arcsearch_acceptance_tol::Float64
    stabilisation_weight::Float64
    cpu_time_limit::Float64
    direct_subproblem_solve::Bool
    exact_arc_search::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sbls_control::sbls_control_type
    convert_control::convert_control_type
end

mutable struct slls_time_type
    total::Float32
    analyse::Float32
    factorize::Float32
    solve::Float32
end

mutable struct slls_inform_type
    status::Cint
    alloc_status::Cint
    factorization_status::Cint
    iter::Cint
    cg_iter::Cint
    obj::Float64
    norm_pg::Float64
    bad_alloc::NTuple{81,Cchar}
    time::slls_time_type
    sbls_inform::sbls_inform_type
    convert_inform::convert_inform_type
end

function slls_initialize(data, control, status)
    @ccall libgalahad_double.slls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{slls_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function slls_read_specfile(control, specfile)
    @ccall libgalahad_double.slls_read_specfile(control::Ptr{slls_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function slls_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.slls_import(control::Ptr{slls_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

function slls_import_without_a(control, data, status, n, m)
    @ccall libgalahad_double.slls_import_without_a(control::Ptr{slls_control_type},
                                                   data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, m::Cint)::Cvoid
end

function slls_reset_control(control, data, status)
    @ccall libgalahad_double.slls_reset_control(control::Ptr{slls_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

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

function slls_information(data, inform, status)
    @ccall libgalahad_double.slls_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{slls_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function slls_terminate(data, control, inform)
    @ccall libgalahad_double.slls_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{slls_control_type},
                                            inform::Ptr{slls_inform_type})::Cvoid
end
