mutable struct fit_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct fit_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
end

function fit_initialize(data, control, status)
    @ccall libgalahad_double.fit_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{fit_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function fit_information(data, inform, status)
    @ccall libgalahad_double.fit_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{fit_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function fit_terminate(data, control, inform)
    @ccall libgalahad_double.fit_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{fit_control_type},
                                           inform::Ptr{fit_inform_type})::Cvoid
end
