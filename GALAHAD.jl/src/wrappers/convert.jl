mutable struct convert_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    transpose::Bool
    sum_duplicates::Bool
    order::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct convert_time_type
    total::Float64
    clock_total::Float64
end

mutable struct convert_inform_type
    status::Cint
    alloc_status::Cint
    duplicates::Cint
    bad_alloc::NTuple{81,Cchar}
    time::convert_time_type
end

function convert_initialize(data, control, status)
    @ccall libgalahad_double.convert_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ref{convert_control_type},
                                                status::Ptr{Cint})::Cvoid
end

function convert_information(data, inform, status)
    @ccall libgalahad_double.convert_information(data::Ptr{Ptr{Cvoid}},
                                                 inform::Ref{convert_inform_type},
                                                 status::Ptr{Cint})::Cvoid
end

function convert_terminate(data, control, inform)
    @ccall libgalahad_double.convert_terminate(data::Ptr{Ptr{Cvoid}},
                                               control::Ref{convert_control_type},
                                               inform::Ref{convert_inform_type})::Cvoid
end
