mutable struct qpa_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    factor::Cint
    max_col::Cint
    max_sc::Cint
    indmin::Cint
    valmin::Cint
    itref_max::Cint
    infeas_check_interval::Cint
    cg_maxit::Cint
    precon::Cint
    nsemib::Cint
    full_max_fill::Cint
    deletion_strategy::Cint
    restore_problem::Cint
    monitor_residuals::Cint
    cold_start::Cint
    sif_file_device::Cint
    infinity::Float64
    feas_tol::Float64
    obj_unbounded::Float64
    increase_rho_g_factor::Float64
    infeas_g_improved_by_factor::Float64
    increase_rho_b_factor::Float64
    infeas_b_improved_by_factor::Float64
    pivot_tol::Float64
    pivot_tol_for_dependencies::Float64
    zero_pivot::Float64
    inner_stop_relative::Float64
    inner_stop_absolute::Float64
    multiplier_tol::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    treat_zero_bounds_as_general::Bool
    solve_qp::Bool
    solve_within_bounds::Bool
    randomize::Bool
    array_syntax_worse_than_do_loop::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    symmetric_linear_solver::NTuple{31,Cchar}
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    each_interval::Bool
    sls_control::sls_control_type
end

mutable struct qpa_time_type
    total::Float64
    preprocess::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    clock_total::Float64
    clock_preprocess::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct qpa_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    major_iter::Cint
    iter::Cint
    cg_iter::Cint
    factorization_status::Cint
    factorization_integer::Int64
    factorization_real::Int64
    nfacts::Cint
    nmods::Cint
    num_g_infeas::Cint
    num_b_infeas::Cint
    obj::Float64
    infeas_g::Float64
    infeas_b::Float64
    merit::Float64
    time::qpa_time_type
    sls_inform::sls_inform_type
end

function qpa_initialize(data, control, status)
    @ccall libgalahad_double.qpa_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{qpa_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function qpa_read_specfile(control, specfile)
    @ccall libgalahad_double.qpa_read_specfile(control::Ref{qpa_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function qpa_import(control, data, status, n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type,
                    A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.qpa_import(control::Ref{qpa_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, H_type::Ptr{Cchar}, H_ne::Cint,
                                        H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                        H_ptr::Ptr{Cint}, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function qpa_reset_control(control, data, status)
    @ccall libgalahad_double.qpa_reset_control(control::Ref{qpa_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function qpa_solve_qp(data, status, n, m, h_ne, H_val, g, f, a_ne, A_val, c_l, c_u, x_l,
                      x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.qpa_solve_qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                          g::Ptr{Float64}, f::Float64, a_ne::Cint,
                                          A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                          c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                          x_u::Ptr{Float64}, x::Ptr{Float64},
                                          c::Ptr{Float64}, y::Ptr{Float64},
                                          z::Ptr{Float64}, x_stat::Ptr{Cint},
                                          c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_l1qp(data, status, n, m, h_ne, H_val, g, f, rho_g, rho_b, a_ne, A_val,
                        c_l, c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.qpa_solve_l1qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                            n::Cint, m::Cint, h_ne::Cint,
                                            H_val::Ptr{Float64}, g::Ptr{Float64},
                                            f::Float64, rho_g::Float64, rho_b::Float64,
                                            a_ne::Cint, A_val::Ptr{Float64},
                                            c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                            x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                            x::Ptr{Float64}, c::Ptr{Float64},
                                            y::Ptr{Float64}, z::Ptr{Float64},
                                            x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function qpa_solve_bcl1qp(data, status, n, m, h_ne, H_val, g, f, rho_g, a_ne, A_val, c_l,
                          c_u, x_l, x_u, x, c, y, z, x_stat, c_stat)
    @ccall libgalahad_double.qpa_solve_bcl1qp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                              n::Cint, m::Cint, h_ne::Cint,
                                              H_val::Ptr{Float64}, g::Ptr{Float64},
                                              f::Float64, rho_g::Float64, a_ne::Cint,
                                              A_val::Ptr{Float64}, c_l::Ptr{Float64},
                                              c_u::Ptr{Float64}, x_l::Ptr{Float64},
                                              x_u::Ptr{Float64}, x::Ptr{Float64},
                                              c::Ptr{Float64}, y::Ptr{Float64},
                                              z::Ptr{Float64}, x_stat::Ptr{Cint},
                                              c_stat::Ptr{Cint})::Cvoid
end

function qpa_information(data, inform, status)
    @ccall libgalahad_double.qpa_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{qpa_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function qpa_terminate(data, control, inform)
    @ccall libgalahad_double.qpa_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{qpa_control_type},
                                           inform::Ref{qpa_inform_type})::Cvoid
end
