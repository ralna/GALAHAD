mutable struct lsrt_control_type
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

mutable struct lsrt_inform_type
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

function lsrt_initialize(data, control, status)
    @ccall libgalahad_double.lsrt_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lsrt_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function lsrt_read_specfile(control, specfile)
    @ccall libgalahad_double.lsrt_read_specfile(control::Ptr{lsrt_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lsrt_import_control(control, data, status)
    @ccall libgalahad_double.lsrt_import_control(control::Ptr{lsrt_control_type},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function lsrt_solve_problem(data, status, m, n, power, weight, x, u, v)
    @ccall libgalahad_double.lsrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, power::Float64,
                                                weight::Float64, x::Ptr{Float64},
                                                u::Ptr{Float64}, v::Ptr{Float64})::Cvoid
end

function lsrt_information(data, inform, status)
    @ccall libgalahad_double.lsrt_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lsrt_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function lsrt_terminate(data, control, inform)
    @ccall libgalahad_double.lsrt_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lsrt_control_type},
                                            inform::Ptr{lsrt_inform_type})::Cvoid
end
