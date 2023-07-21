mutable struct roots_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    tol::Float64
    zero_coef::Float64
    zero_f::Float64
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct roots_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
end

function roots_initialize(data, control, status)
    @ccall libgalahad_double.roots_initialize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{roots_control_type},
                                              status::Ptr{Cint})::Cvoid
end

function roots_information(data, inform, status)
    @ccall libgalahad_double.roots_information(data::Ptr{Ptr{Cvoid}},
                                               inform::Ptr{roots_inform_type},
                                               status::Ptr{Cint})::Cvoid
end

function roots_terminate(data, control, inform)
    @ccall libgalahad_double.roots_terminate(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{roots_control_type},
                                             inform::Ptr{roots_inform_type})::Cvoid
end
