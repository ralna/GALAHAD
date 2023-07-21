mutable struct dps_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    problem::Cint
    print_level::Cint
    new_h::Cint
    taylor_max_degree::Cint
    eigen_min::Float64
    lower::Float64
    upper::Float64
    stop_normal::Float64
    stop_absolute_normal::Float64
    goldfarb::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    problem_file::NTuple{31,Cchar}
    symmetric_linear_solver::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sls_control::sls_control_type
end

mutable struct dps_time_type
    total::Float64
    analyse::Float64
    factorize::Float64
    solve::Float64
    clock_total::Float64
    clock_analyse::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct dps_inform_type
    status::Cint
    alloc_status::Cint
    mod_1by1::Cint
    mod_2by2::Cint
    obj::Float64
    obj_regularized::Float64
    x_norm::Float64
    multiplier::Float64
    pole::Float64
    hard_case::Bool
    bad_alloc::NTuple{81,Cchar}
    time::dps_time_type
    sls_inform::sls_inform_type
end

function dps_initialize(data, control, status)
    @ccall libgalahad_double.dps_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{dps_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function dps_read_specfile(control, specfile)
    @ccall libgalahad_double.dps_read_specfile(control::Ref{dps_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function dps_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.dps_import(control::Ref{dps_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function dps_reset_control(control, data, status)
    @ccall libgalahad_double.dps_reset_control(control::Ref{dps_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function dps_solve_tr_problem(data, status, n, ne, H_val, c, f, radius, x)
    @ccall libgalahad_double.dps_solve_tr_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, ne::Cint, H_val::Ptr{Float64},
                                                  c::Ptr{Float64}, f::Float64,
                                                  radius::Float64, x::Ptr{Float64})::Cvoid
end

function dps_solve_rq_problem(data, status, n, ne, H_val, c, f, power, weight, x)
    @ccall libgalahad_double.dps_solve_rq_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint, ne::Cint, H_val::Ptr{Float64},
                                                  c::Ptr{Float64}, f::Float64,
                                                  power::Float64, weight::Float64,
                                                  x::Ptr{Float64})::Cvoid
end

function dps_resolve_tr_problem(data, status, n, c, f, radius, x)
    @ccall libgalahad_double.dps_resolve_tr_problem(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    c::Ptr{Float64}, f::Float64,
                                                    radius::Float64,
                                                    x::Ptr{Float64})::Cvoid
end

function dps_resolve_rq_problem(data, status, n, c, f, power, weight, x)
    @ccall libgalahad_double.dps_resolve_rq_problem(data::Ptr{Ptr{Cvoid}},
                                                    status::Ptr{Cint}, n::Cint,
                                                    c::Ptr{Float64}, f::Float64,
                                                    power::Float64, weight::Float64,
                                                    x::Ptr{Float64})::Cvoid
end

function dps_information(data, inform, status)
    @ccall libgalahad_double.dps_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{dps_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function dps_terminate(data, control, inform)
    @ccall libgalahad_double.dps_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{dps_control_type},
                                           inform::Ref{dps_inform_type})::Cvoid
end
