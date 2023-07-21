mutable struct glrt_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    itmax::Cint
    stopping_rule::Cint
    freq::Cint
    extra_vectors::Cint
    ritz_printout_device::Cint
    stop_relative::Float64
    stop_absolute::Float64
    fraction_opt::Float64
    rminvr_zero::Float64
    f_0::Float64
    unitm::Bool
    impose_descent::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    print_ritz_values::Bool
    ritz_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
end

mutable struct glrt_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    iter_pass2::Cint
    obj::Float64
    obj_regularized::Float64
    multiplier::Float64
    xpo_norm::Float64
    leftmost::Float64
    negative_curvature::Bool
    hard_case::Bool
end

function glrt_initialize(data, control, status)
    @ccall libgalahad_double.glrt_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ref{glrt_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function glrt_read_specfile(control, specfile)
    @ccall libgalahad_double.glrt_read_specfile(control::Ref{glrt_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function glrt_import_control(control, data, status)
    @ccall libgalahad_double.glrt_import_control(control::Ref{glrt_control_type},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function glrt_solve_problem(data, status, n, power, weight, x, r, vector)
    @ccall libgalahad_double.glrt_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, power::Float64, weight::Float64,
                                                x::Ptr{Float64}, r::Ptr{Float64},
                                                vector::Ptr{Float64})::Cvoid
end

function glrt_information(data, inform, status)
    @ccall libgalahad_double.glrt_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ref{glrt_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function glrt_terminate(data, control, inform)
    @ccall libgalahad_double.glrt_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{glrt_control_type},
                                            inform::Ref{glrt_inform_type})::Cvoid
end
