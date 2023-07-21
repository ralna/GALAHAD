mutable struct ir_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    itref_max::Cint
    acceptable_residual_relative::Float64
    acceptable_residual_absolute::Float64
    required_residual_relative::Float64
    record_residuals::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    prefix::NTuple{31,Cchar}
end

mutable struct ir_inform_type
    status::Cint
    alloc_status::Cint
    bad_alloc::NTuple{81,Cchar}
    norm_initial_residual::Float64
    norm_final_residual::Float64
end

function ir_initialize(data, control, status)
    @ccall libgalahad_double.ir_initialize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{ir_control_type},
                                           status::Ptr{Cint})::Cvoid
end

function ir_information(data, inform, status)
    @ccall libgalahad_double.ir_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{ir_inform_type},
                                            status::Ptr{Cint})::Cvoid
end

function ir_terminate(data, control, inform)
    @ccall libgalahad_double.ir_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{ir_control_type},
                                          inform::Ptr{ir_inform_type})::Cvoid
end
