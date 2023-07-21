mutable struct icfs_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    icfs_vectors::Cint
    shift::Float64
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct icfs_time_type
    total::Float32
    factorize::Float32
    solve::Float32
    clock_total::Float64
    clock_factorize::Float64
    clock_solve::Float64
end

mutable struct icfs_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    mc61_info::NTuple{10,Cint}
    mc61_rinfo::NTuple{15,Float64}
    time::icfs_time_type
end

function icfs_initialize(data, control, status)
    @ccall libgalahad_double.icfs_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{icfs_control_type},
                                             status::Ptr{Cint})::Cvoid
end

function icfs_read_specfile(control, specfile)
    @ccall libgalahad_double.icfs_read_specfile(control::Ptr{icfs_control_type},
                                                specfile::Ptr{Cchar})::Cvoid
end

function icfs_reset_control(control, data, status)
    @ccall libgalahad_double.icfs_reset_control(control::Ptr{icfs_control_type},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Cint})::Cvoid
end

function icfs_factorize_matrix(data, status, n, ptr)
    @ccall libgalahad_double.icfs_factorize_matrix(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                   n::Cint, ptr::Ptr{Cint})::Cvoid
end

function icfs_solve_system(data, status, n, sol, trans)
    @ccall libgalahad_double.icfs_solve_system(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, sol::Ptr{Float64},
                                               trans::Bool)::Cvoid
end

function icfs_information(data, inform, status)
    @ccall libgalahad_double.icfs_information(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{icfs_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

function icfs_terminate(data, control, inform)
    @ccall libgalahad_double.icfs_terminate(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{icfs_control_type},
                                            inform::Ptr{icfs_inform_type})::Cvoid
end
