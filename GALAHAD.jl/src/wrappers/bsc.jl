mutable struct bsc_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    max_col::Cint
    new_a::Cint
    extra_space_s::Cint
    s_also_by_column::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct bsc_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    max_col_a::Cint
    exceeds_max_col::Cint
    time::Float64
    clock_time::Float64
end

function bsc_initialize(data, control, status)
    @ccall libgalahad_double.bsc_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{bsc_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function bsc_information(data, inform, status)
    @ccall libgalahad_double.bsc_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{bsc_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function bsc_terminate(data, control, inform)
    @ccall libgalahad_double.bsc_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{bsc_control_type},
                                           inform::Ref{bsc_inform_type})::Cvoid
end
