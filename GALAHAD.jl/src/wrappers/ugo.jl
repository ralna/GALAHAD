export ugo_control_type

mutable struct ugo_control_type
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    maxit::Cint
    initial_points::Cint
    storage_increment::Cint
    buffer::Cint
    lipschitz_estimate_used::Cint
    next_interval_selection::Cint
    refine_with_newton::Cint
    alive_unit::Cint
    alive_file::NTuple{31,Cchar}
    stop_length::Float64
    small_g_for_newton::Float64
    small_g::Float64
    obj_sufficient::Float64
    global_lipschitz_constant::Float64
    reliability_parameter::Float64
    lipschitz_lower_bound::Float64
    cpu_time_limit::Float64
    clock_time_limit::Float64
    second_derivative_available::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
    ugo_control_type() = new()
end

export ugo_time_type

mutable struct ugo_time_type
    total::Float32
    clock_total::Float64
    ugo_time_type() = new()
end

export ugo_inform_type

mutable struct ugo_inform_type
    status::Cint
    eval_status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    f_eval::Cint
    g_eval::Cint
    h_eval::Cint
    time::ugo_time_type
    ugo_inform_type() = new()
end

export ugo_initialize

function ugo_initialize(data, control, status)
    @ccall libgalahad_double.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{ugo_control_type},
                                            status::Ptr{Cint})::Cvoid
end

export ugo_read_specfile

function ugo_read_specfile(control, specfile)
    @ccall libgalahad_double.ugo_read_specfile(control::Ref{ugo_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

export ugo_import

function ugo_import(control, data, status, x_l, x_u)
    @ccall libgalahad_double.ugo_import(control::Ref{ugo_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                        x_l::Ptr{Float64}, x_u::Ptr{Float64})::Cvoid
end

export ugo_reset_control

function ugo_reset_control(control, data, status)
    @ccall libgalahad_double.ugo_reset_control(control::Ref{ugo_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export ugo_solve_direct

function ugo_solve_direct(data, userdata, status, x, f, g, h, eval_fgh)
    @ccall libgalahad_double.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Cint}, x::Ptr{Float64},
                                              f::Ptr{Float64}, g::Ptr{Float64},
                                              h::Ptr{Float64}, eval_fgh::Ptr{Cvoid})::Cvoid
end

export ugo_solve_reverse

function ugo_solve_reverse(data, status, eval_status, x, f, g, h)
    @ccall libgalahad_double.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               eval_status::Ptr{Cint}, x::Ptr{Float64},
                                               f::Ptr{Float64}, g::Ptr{Float64},
                                               h::Ptr{Float64})::Cvoid
end

export ugo_information

function ugo_information(data, inform, status)
    @ccall libgalahad_double.ugo_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{ugo_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

export ugo_terminate

function ugo_terminate(data, control, inform)
    @ccall libgalahad_double.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{ugo_control_type},
                                           inform::Ref{ugo_inform_type})::Cvoid
end
