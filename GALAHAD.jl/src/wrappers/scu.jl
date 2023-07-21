mutable struct scu_control_type
    f_indexing::Bool
end

mutable struct scu_inform_type
    status::Cint
    alloc_status::Cint
    inertia::NTuple{3,Cint}
end

function scu_information(data, inform, status)
    @ccall libgalahad_double.scu_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{scu_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function scu_terminate(data, control, inform)
    @ccall libgalahad_double.scu_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{scu_control_type},
                                           inform::Ref{scu_inform_type})::Cvoid
end
