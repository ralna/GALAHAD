export hash_control_type

struct hash_control_type
  error::Cint
  out::Cint
  print_level::Cint
  space_critical::Bool
  deallocate_error_fatal::Bool
  prefix::NTuple{31,Cchar}
end

export hash_inform_type

struct hash_inform_type
  status::Cint
  alloc_status::Cint
  bad_alloc::NTuple{81,Cchar}
end

export hash_initialize_s

function hash_initialize_s(nchar, length, data, control, inform)
  @ccall libgalahad_single.hash_initialize_s(nchar::Cint, length::Cint,
                                             data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{hash_control_type},
                                             inform::Ptr{hash_inform_type})::Cvoid
end

export hash_initialize

function hash_initialize(nchar, length, data, control, inform)
  @ccall libgalahad_double.hash_initialize(nchar::Cint, length::Cint, data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{hash_control_type},
                                           inform::Ptr{hash_inform_type})::Cvoid
end

export hash_information_s

function hash_information_s(data, inform, status)
  @ccall libgalahad_single.hash_information_s(data::Ptr{Ptr{Cvoid}},
                                              inform::Ptr{hash_inform_type},
                                              status::Ptr{Cint})::Cvoid
end

export hash_information

function hash_information(data, inform, status)
  @ccall libgalahad_double.hash_information(data::Ptr{Ptr{Cvoid}},
                                            inform::Ptr{hash_inform_type},
                                            status::Ptr{Cint})::Cvoid
end

export hash_terminate_s

function hash_terminate_s(data, control, inform)
  @ccall libgalahad_single.hash_terminate_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{hash_control_type},
                                            inform::Ptr{hash_inform_type})::Cvoid
end

export hash_terminate

function hash_terminate(data, control, inform)
  @ccall libgalahad_double.hash_terminate(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{hash_control_type},
                                          inform::Ptr{hash_inform_type})::Cvoid
end
