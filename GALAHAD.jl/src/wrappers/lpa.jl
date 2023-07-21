mutable struct lpa_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    maxit::Cint
    max_iterative_refinements::Cint
    min_real_factor_size::Cint
    min_integer_factor_size::Cint
    random_number_seed::Cint
    sif_file_device::Cint
    qplib_file_device::Cint
    infinity::Float64
    tol_data::Float64
    feas_tol::Float64
    relative_pivot_tolerance::Float64
    growth_limit::Float64
    zero_tolerance::Float64
    change_tolerance::Float64
    identical_bounds_tol::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    scale::Bool
    dual::Bool
    warm_start::Bool
    steepest_edge::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    generate_qplib_file::Bool
    sif_file_name::NTuple{31,Cchar}
    qplib_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
end

mutable struct lpa_time_type
    total::Float64
    preprocess::Float64
    clock_total::Float64
    clock_preprocess::Float64
end

mutable struct lpa_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    la04_job::Cint
    la04_job_info::Cint
    obj::Float64
    primal_infeasibility::Float64
    feasible::Bool
    RINFO::NTuple{40,Float64}
    time::lpa_time_type
    rpd_inform::rpd_inform_type
end

function lpa_initialize(data, control, status)
    @ccall libgalahad_double.lpa_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{lpa_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function lpa_read_specfile(control, specfile)
    @ccall libgalahad_double.lpa_read_specfile(control::Ref{lpa_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function lpa_import(control, data, status, n, m, A_type, A_ne, A_row, A_col, A_ptr)
    @ccall libgalahad_double.lpa_import(control::Ref{lpa_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        m::Cint, A_type::Ptr{Cchar}, A_ne::Cint,
                                        A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                        A_ptr::Ptr{Cint})::Cvoid
end

function lpa_reset_control(control, data, status)
    @ccall libgalahad_double.lpa_reset_control(control::Ref{lpa_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function lpa_solve_lp(data, status, n, m, g, f, a_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                      x_stat, c_stat)
    @ccall libgalahad_double.lpa_solve_lp(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                          m::Cint, g::Ptr{Float64}, f::Float64,
                                          a_ne::Cint, A_val::Ptr{Float64},
                                          c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                          x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                          x::Ptr{Float64}, c::Ptr{Float64},
                                          y::Ptr{Float64}, z::Ptr{Float64},
                                          x_stat::Ptr{Cint}, c_stat::Ptr{Cint})::Cvoid
end

function lpa_information(data, inform, status)
    @ccall libgalahad_double.lpa_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{lpa_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function lpa_terminate(data, control, inform)
    @ccall libgalahad_double.lpa_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{lpa_control_type},
                                           inform::Ref{lpa_inform_type})::Cvoid
end
