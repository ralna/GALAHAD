mutable struct qpb_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    itref_max::Cint
    cg_maxit::Cint
    indicator_type::Cint
    restore_problem::Cint
    extrapolate::Cint
    path_history::Cint
    factor::Cint
    max_col::Cint
    indmin::Cint
    valmin::Cint
    infeas_max::Cint
    precon::Cint
    nsemib::Cint
    path_derivatives::Cint
    fit_order::Cint
    sif_file_device::Cint
    infinity::Float64
    stop_p::Float64
    stop_d::Float64
    stop_c::Float64
    theta_d::Float64
    theta_c::Float64
    beta::Float64
    prfeas::Float64
    dufeas::Float64
    muzero::Float64
    reduce_infeas::Float64
    obj_unbounded::Float64
    pivot_tol::Float64
    pivot_tol_for_dependencies::Float64
    zero_pivot::Float64
    identical_bounds_tol::Float64
    inner_stop_relative::Float64
    inner_stop_absolute::Float64
    initial_radius::Float64
    mu_min::Float64
    inner_fraction_opt::Float64
    indicator_tol_p::Float64
    indicator_tol_pd::Float64
    indicator_tol_tapia::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    remove_dependencies::Bool
    treat_zero_bounds_as_general::Bool
    center::Bool
    primal::Bool
    puiseux::Bool
    feasol::Bool
    array_syntax_worse_than_do_loop::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    lsqp_control::lsqp_control_type
    fdc_control::fdc_control_type
    sbls_control::sbls_control_type
    gltr_control::gltr_control_type
    fit_control::fit_control_type
end

mutable struct qpb_time_type
    total::Float64
    preprocess::Float64
    find_dependent::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    phase1_total::Float64
    phase1_analyse::Float64
    phase1_factorize::Float64
    phase1_solve::Float64
    clock_total::Float64
    clock_preprocess::Float64
    clock_find_dependent::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
    clock_phase1_total::Float64
    clock_phase1_analyse::Float64
    clock_phase1_factorize::Float64
    clock_phase1_solve::Float64
end

mutable struct qpb_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    cg_iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    nbacts::Cint
    nmods::Cint
    obj::Float64
    non_negligible_pivot::Float64
    feasible::Bool
    time::qpb_time_type
    lsqp_inform::lsqp_inform_type
    fdc_inform::fdc_inform_type
    sbls_inform::sbls_inform_type
    gltr_inform::gltr_inform_type
    fit_inform::fit_inform_type
end

function qpb_initialize(data, control, status)
    @ccall libgalahad_double.qpb_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{qpb_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function qpb_read_specfile(control, specfile)
    @ccall libgalahad_double.qpb_read_specfile(control::Ptr{qpb_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function qpb_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.qpb_import(control::Ptr{qpb_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function qpb_reset_control(control, data, status)
    @ccall libgalahad_double.qpb_reset_control(control::Ptr{qpb_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function qpb_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.qpb_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                          A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, x::Ptr{Float64},
                                          c::Ptr{Float64}, y::Ptr{Float64},
                                          z::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function qpb_information(data, inform, status)
    @ccall libgalahad_double.qpb_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{qpb_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function qpb_terminate(data, control, inform)
    @ccall libgalahad_double.qpb_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{qpb_control_type},
                                           inform::Ptr{qpb_inform_type})::Cvoid
end
