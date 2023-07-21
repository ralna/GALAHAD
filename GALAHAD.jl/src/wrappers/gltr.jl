mutable struct gltr_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    itmax::Cint
    Lanczos_itmax::Cint
    extra_vectors::Cint
    ritz_printout_device::Cint
    stop_relative::Float64
    stop_absolute::Float64
    fraction_opt::Float64
    f_min::Float64
    rminvr_zero::Float64
    f_0::Float64
    unitm::Bool
    steihaug_toint::Bool
    boundary::Bool
    equality_problem::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    print_ritz_values::Bool
    ritz_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
end

mutable struct gltr_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    iter::Cint
    iter_pass2::Cint
    obj::Float64
    multiplier::Float64
    mnormx::Float64
    piv::Float64
    curv::Float64
    rayleigh::Float64
    leftmost::Float64
    negative_curvature::Bool
    hard_case::Bool
end

function gltr_initialize(data, control, status)
    @ccall libgalahad_double.gltr_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gltr_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function gltr_read_specfile(control, specfile)
    @ccall libgalahad_double.gltr_read_specfile(control::Ptr{gltr_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function gltr_import_control(control, data, status)
    @ccall libgalahad_double.gltr_import_control(control::Ptr{gltr_control_type},
                                                 data::Ptr{Ptr{Cvoid}},
                                                 status::Ptr{Cint})::Cvoid
end

function gltr_solve_problem(data, status, n, radius, x, r, vector)
    @ccall libgalahad_double.gltr_solve_problem(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                n::Cint, radius::Float64, x::Ptr{Float64},
                                                r::Ptr{Float64},
                                                vector::Ptr{Float64})::Cvoid
end

function gltr_information(data, inform, status)
    @ccall libgalahad_double.gltr_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{gltr_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function gltr_terminate(data, control, inform)
    @ccall libgalahad_double.gltr_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{gltr_control_type},
                                            inform::Ptr{gltr_inform_type})::Cvoid
end
