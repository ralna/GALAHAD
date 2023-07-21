mutable struct sec_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    h_initial::Float64
    update_skip_tol::Float64
    prefix::NTuple{31,Cchar}
end

mutable struct sec_inform_type
    status::Cint
end

function sec_initialize(control, status)
    @ccall libgalahad_double.sec_initialize(control::Ptr{sec_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function sec_information(data, inform, status)
    @ccall libgalahad_double.sec_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{sec_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function sec_terminate(data, control, inform)
    @ccall libgalahad_double.sec_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{sec_control_type},
                                           inform::Ptr{sec_inform_type})::Cvoid
end
