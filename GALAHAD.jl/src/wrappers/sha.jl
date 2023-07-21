mutable struct sha_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    approximation_algorithm::Cint
    dense_linear_solver::Cint
    max_sparse_degree::Cint
    extra_differences::Cint
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct sha_inform_type
    status::Cint
    alloc_status::Cint
    max_degree::Cint
    differences_needed::Cint
    max_reduced_degree::Cint
    bad_row::Cint
    bad_alloc::NTuple{81,Cchar}
end

function sha_initialize(data, control, status)
    @ccall libgalahad_double.sha_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{sha_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function sha_information(data, inform, status)
    @ccall libgalahad_double.sha_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ref{sha_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function sha_terminate(data, control, inform)
    @ccall libgalahad_double.sha_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ref{sha_control_type},
                                           inform::Ref{sha_inform_type})::Cvoid
end
