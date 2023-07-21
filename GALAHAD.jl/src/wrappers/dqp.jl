mutable struct dqp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    dual_starting_point::Cint
    maxit::Cint
    max_sc::Cint
    cauchy_only::Cint
    arc_search_maxit::Cint
    cg_maxit::Cint
    explore_optimal_subspace::Cint
    restore_problem::Cint
    sif_file_device::Cint
    qplib_file_device::Cint
    rho::Float64
    infinity::Float64
    stop_abs_p::Float64
    stop_rel_p::Float64
    stop_abs_d::Float64
    stop_rel_d::Float64
    stop_abs_c::Float64
    stop_rel_c::Float64
    stop_cg_relative::Float64
    stop_cg_absolute::Float64
    cg_zero_curvature::Float64
    max_growth::Float64
    identical_bounds_tol::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    initial_perturbation::Float64
    perturbation_reduction::Float64
    final_perturbation::Float64
    factor_optimal_matrix::Bool
    remove_dependencies::Bool
    treat_zero_bounds_as_general::Bool
    exact_arc_search::Bool
    subspace_direct::Bool
    subspace_alternate::Bool
    subspace_arc_search::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    generate_qplib_file::Bool
    symmetric_linear_solver::NTuple{31,Cchar}
    definite_linear_solver::NTuple{31,Cchar}
    unsymmetric_linear_solver::NTuple{31,Cchar}
    sif_file_name::NTuple{31,Cchar}
    qplib_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    fdc_control::fdc_control_type
    sls_control::sls_control_type
    sbls_control::sbls_control_type
    gltr_control::gltr_control_type
end

mutable struct dqp_time_type
    total::Float64
    preprocess::Float64
    find_dependent::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    search::Float64
    clock_total::Float64
    clock_preprocess::Float64
    clock_find_dependent::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
    clock_search::Float64
end

mutable struct dqp_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    cg_iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    threads::Cint
    obj::Float64
    primal_infeasibility::Float64
    dual_infeasibility::Float64
    complementary_slackness::Float64
    non_negligible_pivot::Float64
    feasible::Bool
    checkpointsIter::NTuple{16,Cint}
    checkpointsTime::NTuple{16,Float64}
    time::dqp_time_type
    fdc_inform::fdc_inform_type
    sls_inform::sls_inform_type
    sbls_inform::sbls_inform_type
    gltr_inform::gltr_inform_type
    scu_status::Cint
    scu_inform::scu_inform_type
    rpd_inform::rpd_inform_type
end

function dqp_initialize(data, control, status)
    @ccall libgalahad_double.dqp_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{dqp_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function dqp_read_specfile(control, specfile)
    @ccall libgalahad_double.dqp_read_specfile(control::Ptr{dqp_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function dqp_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.dqp_import(control::Ptr{dqp_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function dqp_reset_control(control, data, status)
    @ccall libgalahad_double.dqp_reset_control(control::Ptr{dqp_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function dqp_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.dqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                          A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, x::Ptr{Float64},
                                          c::Ptr{Float64}, y::Ptr{Float64},
                                          z::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function dqp_solve_sldqp(data, status, n, m, w, x0, g, f, a_ne, A_val, c_l, c_u, x_l, x_u,
                         x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.dqp_solve_sldqp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                             n::Cint, m::Cint, w::Ptr{Float64},
                                             x0::Ptr{Float64}, g::Ptr{Float64},
                                             f::Float64, a_ne::Cint, A_val::Ptr{Float64},
                                             c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                             x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                             x::Ptr{Float64}, c::Ptr{Float64},
                                             y::Ptr{Float64}, z::Ptr{Float64},
                                             x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function dqp_information(data, inform, status)
    @ccall libgalahad_double.dqp_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{dqp_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function dqp_terminate(data, control, inform)
    @ccall libgalahad_double.dqp_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{dqp_control_type},
                                           inform::Ptr{dqp_inform_type})::Cvoid
end
