add_library(mergenvision_tracking STATIC
    continuous_iou.cpp
    kalman_filter.cpp
    linear_assignment.cpp
    byte_tracker.cpp
    multi_source_tracker.cpp
    evidence_writer.cpp
)

target_include_directories(mergenvision_tracking PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(mergenvision_tracking PUBLIC cxx_std_20)
set_target_properties(mergenvision_tracking PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(test_tracking_core
    tests/test_tracking_core.cpp
)

target_link_libraries(test_tracking_core mergenvision_tracking)
target_compile_features(test_tracking_core PUBLIC cxx_std_20)

add_executable(test_evidence_writer tests/test_evidence_writer.cpp)
target_link_libraries(test_evidence_writer mergenvision_tracking)
target_compile_features(test_evidence_writer PUBLIC cxx_std_20)

# Sanitized variant for Sprint 06 core gate.
add_executable(test_tracking_core_asan
    tests/test_tracking_core.cpp
)

target_compile_options(test_tracking_core_asan PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer -g
)
target_link_options(test_tracking_core_asan PRIVATE
    -fsanitize=address,undefined
)
target_link_libraries(test_tracking_core_asan mergenvision_tracking)
target_compile_features(test_tracking_core_asan PUBLIC cxx_std_20)
