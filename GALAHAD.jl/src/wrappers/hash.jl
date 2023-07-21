mutable struct hash_control_type
    error::Cint
    out::Cint
    print_level::Cint
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct hash_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
end

function hash_initialize(nchar, length, data, control, inform)
    @ccall libgalahad_double.hash_initialize(nchar::Cint, length::Cint,
                                             data::Ptr{Ptr{Cvoid}},
                                             control::Ref{hash_control_type},
                                             inform::Ref{hash_inform_type})::Cvoid
end

function hash_information(data, inform, status)
    @ccall libgalahad_double.hash_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{hash_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function hash_terminate(data, control, inform)
    @ccall libgalahad_double.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{hash_control_type},
                                            inform::Ref{hash_inform_type})::Cvoid
end
