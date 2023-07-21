mutable struct cqp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    infeas_max::Cint
    muzero_fixed::Cint
    restore_problem::Cint
    indicator_type::Cint
    arc::Cint
    series_order::Cint
    sif_file_device::Cint
    qplib_file_device::Cint
    infinity::Float64
    stop_abs_p::Float64
    stop_rel_p::Float64
    stop_abs_d::Float64
    stop_rel_d::Float64
    stop_abs_c::Float64
    stop_rel_c::Float64
    perturb_h::Float64
    prfeas::Float64
    dufeas::Float64
    muzero::Float64
    tau::Float64
    gamma_c::Float64
    gamma_f::Float64
    reduce_infeas::Float64
    obj_unbounded::Float64
    potential_unbounded::Float64
    identical_bounds_tol::Float64
    mu_lunge::Float64
    indicator_tol_p::Float64
    indicator_tol_pd::Float64
    indicator_tol_tapia::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    remove_dependencies::Bool
    treat_zero_bounds_as_general::Bool
    treat_separable_as_general::Bool
    just_feasible::Bool
    getdua::Bool
    puiseux::Bool
    every_order::Bool
    feasol::Bool
    balance_initial_complentarity::Bool
    crossover::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    generate_qplib_file::Bool
    sif_file_name::NTuple{31,Cchar}
    qplib_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    fdc_control::fdc_control_type
    sbls_control::sbls_control_type
    fit_control::fit_control_type
    roots_control::roots_control_type
    cro_control::cro_control_type
end

mutable struct cqp_time_type
    total::Float64
    preprocess::Float64
    find_dependent::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    clock_total::Float64
    clock_preprocess::Float64
    clock_find_dependent::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct cqp_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    nbacts::Cint
    threads::Cint
    obj::Float64
    primal_infeasibility::Float64
    dual_infeasibility::Float64
    complementary_slackness::Float64
    init_primal_infeasibility::Float64
    init_dual_infeasibility::Float64
    init_complementary_slackness::Float64
    potential::Float64
    non_negligible_pivot::Float64
    feasible::Bool
    checkpointsIter::NTuple{16,Cint}
    checkpointsTime::NTuple{16,Float64}
    time::cqp_time_type
    fdc_inform::fdc_inform_type
    sbls_inform::sbls_inform_type
    fit_inform::fit_inform_type
    roots_inform::roots_inform_type
    cro_inform::cro_inform_type
    rpd_inform::rpd_inform_type
end

function cqp_initialize(data, control, status)
    @ccall libgalahad_double.cqp_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{cqp_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function cqp_read_specfile(control, specfile)
    @ccall libgalahad_double.cqp_read_specfile(control::Ref{cqp_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function cqp_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.cqp_import(control::Ref{cqp_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function cqp_reset_control(control, data, status)
    @ccall libgalahad_double.cqp_reset_control(control::Ref{cqp_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function cqp_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.cqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                          A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, x::Ptr{Float64},
                                          c::Ptr{Float64}, y::Ptr{Float64},
                                          z::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function cqp_solve_sldqp(data, status, n, m, w, x0, g, f, a_ne, A_val, c_l, c_u, x_l, x_u,
                         x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.cqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, w::Ptr{Float64},
                                             x0::Ptr{Float64}, g::Ptr{Float64},
                                             f::Float64, a_ne::Cint, A_val::Ptr{Float64},
                                             c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                             x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                             x::Ptr{Float64}, c::Ptr{Float64},
                                             y::Ptr{Float64}, z::Ptr{Float64},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function cqp_information(data, inform, status)
    @ccall libgalahad_double.cqp_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{cqp_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function cqp_terminate(data, control, inform)
    @ccall libgalahad_double.cqp_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{cqp_control_type},
                                           inform::Ref{cqp_inform_type})::Cvoid
end
