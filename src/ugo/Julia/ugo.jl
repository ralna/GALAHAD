using Libdl

libgalahad_c_single = "$(ENV["GALAHAD"])/builddir/libgalahad_c_single.$dlext"
libgalahad_c_double = "$(ENV["GALAHAD"])/builddir/libgalahad_c_double.$dlext"

Base.@kwdef mutable struct ugo_control_type{T}
    error::Int32 = 0
    out::Int32 = 0
    print_level::Int32 = 0
    start_print::Int32 = 0
    stop_print::Int32 = 0
    print_gap::Int32 = 0
    maxit::Int32 = 0
    initial_points::Int32 = 0
    storage_increment::Int32 = 0
    buffer::Int32 = 0
    lipschitz_estimate_used::Int32 = 0
    next_interval_selection::Int32 = 0
    refine_with_newton::Int32 = 0
    alive_unit::Int32 = 0
    alive_file::NTuple{31,Cchar} = ntuple(x -> Cchar(' '), 31)
    stop_length::T = zero(T)
    small_g_for_newton::T = zero(T)
    small_g::T = zero(T)
    obj_sufficient::T = zero(T)
    global_lipschitz_constant::T = zero(T)
    reliability_parameter::T = zero(T)
    lipschitz_lower_bound::T = zero(T)
    cpu_time_limit::T = zero(T)
    clock_time_limit::T = zero(T)
    space_critical::Bool = false
    deallocate_error_fatal::Bool = false
    prefix::NTuple{31,Cchar} = ntuple(x -> Cchar(' '), 31)
end

Base.@kwdef mutable struct ugo_time_type{T}
    total::Float32 = zero(Float32)
    clock_total::T = zero(T)
end

Base.@kwdef mutable struct ugo_inform_type{T}
    status::Int32 = 0
    eval_status::Int32 = 0
    alloc_status::Int32 = 0
    bad_alloc::NTuple{81,Cchar} = ntuple(x -> Cchar(' '), 81)
    iter::Int32 = 0
    f_eval::Int32 = 0
    g_eval::Int32 = 0
    h_eval::Int32 = 0
    time::ugo_time_type{T} = ugo_time_type{T}()
end

# single precision

function ugo_initialize(data, control::ugo_control_type{Float32}, status)
    @ccall libgalahad_c_single.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{ugo_control_type{Float32}},
                                            status::Ptr{Int32})::Cvoid
end

function ugo_read_specfile(control::ugo_control_type{Float32}, specfile)
    @ccall libgalahad_c_single.ugo_read_specfile(control::Ref{ugo_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function ugo_import(control::ugo_control_type{Float32}, data, status, x_l::Float32, x_u::Float32)
    @ccall libgalahad_c_single.ugo_import(control::Ref{ugo_control_type{Float32}}, data::Ptr{Ptr{Cvoid}},
                                          status::Ptr{Int32}, x_l::Ref{Float32},
                                          x_u::Ref{Float32})::Cvoid
end

function ugo_reset_control(control::ugo_control_type{Float32}, data, status)
    @ccall libgalahad_c_single.ugo_reset_control(control::Ref{ugo_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function ugo_solve_direct(data, userdata, status, x::Float32, f::Float32, g::Float32, h::Float32, eval_fgh)
    @ccall libgalahad_c_single.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, x::Ref{Float32},
                                              f::Ref{Float32}, g::Ref{Float32},
                                              h::Ref{Float32}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_reverse(data, status, eval_status, x::Float32, f::Float32, g::Float32, h::Float32)
    @ccall libgalahad_c_single.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                               eval_status::Ptr{Int32}, x::Ref{Float32},
                                               f::Ref{Float32}, g::Ref{Float32},
                                               h::Ref{Float32})::Cvoid
end

function ugo_information(data, inform::ugo_inform_type{Float32}, status)
    @ccall libgalahad_c_single.ugo_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{ugo_inform_type{Float32}},
                                             status::Ptr{Int32})::Cvoid
end

function ugo_terminate(data, control::ugo_control_type{Float32}, inform::ugo_inform_type{Float32})
    @ccall libgalahad_c_single.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{ugo_control_type{Float32}},
                                           inform::Ref{ugo_inform_type{Float32}})::Cvoid
end

# double precision

function ugo_initialize(data, control::ugo_control_type{Float64}, status)
    @ccall libgalahad_c_double.ugo_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{ugo_control_type{Float64}},
                                            status::Ptr{Int32})::Cvoid
end

function ugo_read_specfile(control::ugo_control_type{Float64}, specfile)
    @ccall libgalahad_c_double.ugo_read_specfile(control::Ref{ugo_control_type{Float64}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function ugo_import(control::ugo_control_type{Float64}, data, status, x_l::Float64, x_u::Float64)
    @ccall libgalahad_c_double.ugo_import(control::Ref{ugo_control_type{Float64}}, data::Ptr{Ptr{Cvoid}},
                                        status::Ptr{Int32}, x_l::Ref{Float64},
                                        x_u::Ref{Float64})::Cvoid
end

function ugo_reset_control(control::ugo_control_type{Float64}, data, status)
    @ccall libgalahad_c_double.ugo_reset_control(control::Ref{ugo_control_type{Float64}},
                                               data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function ugo_solve_direct(data, userdata, status, x::Float64, f::Float64, g::Float64, h::Float64, eval_fgh)
    @ccall libgalahad_c_double.ugo_solve_direct(data::Ptr{Ptr{Cvoid}}, userdata::Ptr{Cvoid},
                                              status::Ptr{Int32}, x::Ref{Float64},
                                              f::Ref{Float64}, g::Ref{Float64},
                                              h::Ref{Float64}, eval_fgh::Ptr{Cvoid})::Cvoid
end

function ugo_solve_reverse(data, status, eval_status, x::Float64, f::Float64, g::Float64, h::Float64)
    @ccall libgalahad_c_double.ugo_solve_reverse(data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32},
                                               eval_status::Ptr{Int32}, x::Ref{Float64},
                                               f::Ref{Float64}, g::Ref{Float64},
                                               h::Ref{Float64})::Cvoid
end

function ugo_information(data, inform::ugo_inform_type{Float64}, status)
    @ccall libgalahad_c_double.ugo_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{ugo_inform_type{Float64}},
                                             status::Ptr{Int32})::Cvoid
end

function ugo_terminate(data, control::ugo_control_type{Float64}, inform::ugo_inform_type{Float64})
    @ccall libgalahad_c_double.ugo_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{ugo_control_type{Float64}},
                                           inform::Ref{ugo_inform_type{Float64}})::Cvoid
end
