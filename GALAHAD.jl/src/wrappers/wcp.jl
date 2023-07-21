mutable struct wcp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    initial_point::Cint
    factor::Cint
    max_col::Cint
    indmin::Cint
    valmin::Cint
    itref_max::Cint
    infeas_max::Cint
    perturbation_strategy::Cint
    restore_problem::Cint
    infinity::Float64
    stop_p::Float64
    stop_d::Float64
    stop_c::Float64
    prfeas::Float64
    dufeas::Float64
    mu_target::Float64
    mu_accept_fraction::Float64
    mu_increase_factor::Float64
    required_infeas_reduction::Float64
    implicit_tol::Float64
    pivot_tol::Float64
    pivot_tol_for_dependencies::Float64
    zero_pivot::Float64
    perturb_start::Float64
    alpha_scale::Float64
    identical_bounds_tol::Float64
    reduce_perturb_factor::Float64
    reduce_perturb_multiplier::Float64
    insufficiently_feasible::Float64
    perturbation_small::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    remove_dependencies::Bool
    treat_zero_bounds_as_general::Bool
    just_feasible::Bool
    balance_initial_complementarity::Bool
    use_corrector::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    record_x_status::Bool
    record_c_status::Bool
    prefix::NTuple{31,Cchar}
    fdc_control::fdc_control_type
    sbls_control::sbls_control_type
end

mutable struct wcp_time_type
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

mutable struct wcp_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    c_implicit::Cint
    x_implicit::Cint
    y_implicit::Cint
    z_implicit::Cint
    obj::Float64
    mu_final_target_max::Float64
    non_negligible_pivot::Float64
    feasible::Bool
    time::wcp_time_type
    fdc_inform::fdc_inform_type
    sbls_inform::sbls_inform_type
end

function wcp_initialize(data, control, status)
    @ccall libgalahad_double.wcp_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{wcp_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function wcp_read_specfile(control, specfile)
    @ccall libgalahad_double.wcp_read_specfile(control::Ref{wcp_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function wcp_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.wcp_import(control::Ref{wcp_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function wcp_reset_control(control, data, status)
    @ccall libgalahad_double.wcp_reset_control(control::Ref{wcp_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function wcp_find_wcp(data, status, n, m, g, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y_l,
                      y_u, z_l, z_u, x_stat, c_stat)
    @ccall libgalahad_double.wcp_find_wcp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float64}, a_ne::Cint,
                                          A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, x::Ptr{Float64},
                                          c::Ptr{Float64}, y_l::Ptr{Float64},
                                          y_u::Ptr{Float64}, z_l::Ptr{Float64},
                                          z_u::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function wcp_information(data, inform, status)
    @ccall libgalahad_double.wcp_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{wcp_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function wcp_terminate(data, control, inform)
    @ccall libgalahad_double.wcp_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{wcp_control_type},
                                           inform::Ref{wcp_inform_type})::Cvoid
end
