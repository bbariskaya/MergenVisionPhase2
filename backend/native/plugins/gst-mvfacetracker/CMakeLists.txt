find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0)

find_package(CUDAToolkit REQUIRED)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-9.0")
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_ROOT}/sources/includes")

add_library(gstmvfacetracker SHARED
    gstmvfacetracker.cpp
    tracklet_meta.cpp
)

target_include_directories(gstmvfacetracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tracking
    ${GSTREAMER_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDES}
)

target_link_libraries(gstmvfacetracker
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    mergenvision_tracking
    nvds_meta
    nvdsgst_meta
    nvds_utils
)

set_target_properties(gstmvfacetracker PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gst-plugins
)
