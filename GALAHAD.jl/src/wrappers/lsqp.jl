mutable struct lsqp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    factor::Cint
    max_col::Cint
    indmin::Cint
    valmin::Cint
    itref_max::Cint
    infeas_max::Cint
    muzero_fixed::Cint
    restore_problem::Cint
    indicator_type::Cint
    extrapolate::Cint
    path_history::Cint
    path_derivatives::Cint
    fit_order::Cint
    sif_file_device::Cint
    infinity::Float64
    stop_p::Float64
    stop_d::Float64
    stop_c::Float64
    prfeas::Float64
    dufeas::Float64
    muzero::Float64
    reduce_infeas::Float64
    potential_unbounded::Float64
    pivot_tol::Float64
    pivot_tol_for_dependencies::Float64
    zero_pivot::Float64
    identical_bounds_tol::Float64
    mu_min::Float64
    indicator_tol_p::Float64
    indicator_tol_pd::Float64
    indicator_tol_tapia::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    remove_dependencies::Bool
    treat_zero_bounds_as_general::Bool
    just_feasible::Bool
    getdua::Bool
    puiseux::Bool
    feasol::Bool
    balance_initial_complentarity::Bool
    use_corrector::Bool
    array_syntax_worse_than_do_loop::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    fdc_control::fdc_control_type
    sbls_control::sbls_control_type
end

mutable struct lsqp_time_type
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

mutable struct lsqp_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    nbacts::Cint
    obj::Float64
    potential::Float64
    non_negligible_pivot::Float64
    feasible::Bool
    time::lsqp_time_type
    fdc_inform::fdc_inform_type
    sbls_inform::sbls_inform_type
end

function lsqp_initialize(data, control, status)
    @ccall libgalahad_double.lsqp_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lsqp_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function lsqp_read_specfile(control, specfile)
    @ccall libgalahad_double.lsqp_read_specfile(control::Ptr{lsqp_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lsqp_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.lsqp_import(control::Ptr{lsqp_control_type},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                         m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                         A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                         A_ptr::Ptr{Cint})::Cvoid
end

function lsqp_reset_control(control, data, status)
    @ccall libgalahad_double.lsqp_reset_control(control::Ptr{lsqp_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function lsqp_solve_qp(data, status, n, m, w, x0, g, f, a_ne, A_val, c_l, c_u, x_l, x_u, x,
                       c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.lsqp_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                           n::Cint, m::Cint, w::Ptr{Float64},
                                           x0::Ptr{Float64}, g::Ptr{Float64}, f::Float64,
                                           a_ne::Cint, A_val::Ptr{Float64},
                                           c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                           x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                           x::Ptr{Float64}, c::Ptr{Float64},
                                           y::Ptr{Float64}, z::Ptr{Float64},
                                           x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function lsqp_information(data, inform, status)
    @ccall libgalahad_double.lsqp_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lsqp_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function lsqp_terminate(data, control, inform)
    @ccall libgalahad_double.lsqp_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lsqp_control_type},
                                            inform::Ptr{lsqp_inform_type})::Cvoid
end
