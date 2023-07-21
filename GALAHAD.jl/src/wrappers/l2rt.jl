mutable struct l2rt_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    itmin::Cint
    itmax::Cint
    bitmax::Cint
    extra_vectors::Cint
    stopping_rule::Cint
    freq::Cint
    stop_relative::Float64
    stop_absolute::Float64
    fraction_opt::Float64
    time_limit::Float64
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct l2rt_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    iter_pass2::Cint
    biters::Cint
    biter_min::Cint
    biter_max::Cint
    obj::Float64
    multiplier::Float64
    x_norm::Float64
    r_norm::Float64
    Atr_norm::Float64
    biter_mean::Float64
end

function l2rt_initialize(data, control, status)
    @ccall libgalahad_double.l2rt_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{l2rt_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function l2rt_read_specfile(control, specfile)
    @ccall libgalahad_double.l2rt_read_specfile(control::Ref{l2rt_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function l2rt_import_control(control, data, status)
    @ccall libgalahad_double.l2rt_import_control(control::Ref{l2rt_control_type},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function l2rt_solve_problem(data, status, m, n, power, weight, shift, x, u, v)
    @ccall libgalahad_double.l2rt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float64,
                                                weight::Float64, shift::Float64,
                                                x::Ptr{Float64}, u::Ptr{Float64},
                                                v::Ptr{Float64})::Cvoid
end

function l2rt_information(data, inform, status)
    @ccall libgalahad_double.l2rt_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{l2rt_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function l2rt_terminate(data, control, inform)
    @ccall libgalahad_double.l2rt_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{l2rt_control_type},
                                            inform::Ref{l2rt_inform_type})::Cvoid
end
