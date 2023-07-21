mutable struct lstr_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    itmin::Cint
    itmax::Cint
    itmax_on_boundary::Cint
    bitmax::Cint
    extra_vectors::Cint
    stop_relative::Float64
    stop_absolute::Float64
    fraction_opt::Float64
    time_limit::Float64
    steihaug_toint::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct lstr_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    iter_pass2::Cint
    biters::Cint
    biter_min::Cint
    biter_max::Cint
    multiplier::Float64
    x_norm::Float64
    r_norm::Float64
    Atr_norm::Float64
    biter_mean::Float64
end

function lstr_initialize(data, control, status)
    @ccall libgalahad_double.lstr_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{lstr_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function lstr_read_specfile(control, specfile)
    @ccall libgalahad_double.lstr_read_specfile(control::Ptr{lstr_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function lstr_import_control(control, data, status)
    @ccall libgalahad_double.lstr_import_control(control::Ptr{lstr_control_type},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function lstr_solve_problem(data, status, m, n, radius, x, u, v)
    @ccall libgalahad_double.lstr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                m::Cint, n::Cint, radius::Float64,
                                                x::Ptr{Float64}, u::Ptr{Float64},
                                                v::Ptr{Float64})::Cvoid
end

function lstr_information(data, inform, status)
    @ccall libgalahad_double.lstr_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{lstr_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function lstr_terminate(data, control, inform)
    @ccall libgalahad_double.lstr_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{lstr_control_type},
                                            inform::Ptr{lstr_inform_type})::Cvoid
end
