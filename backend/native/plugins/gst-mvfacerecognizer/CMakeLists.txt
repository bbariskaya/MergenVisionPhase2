find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0)

find_package(CUDAToolkit REQUIRED)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-9.0")
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_ROOT}/sources/includes")

add_library(gstmvfacerecognizer SHARED
    gstmvfacerecognizer.cpp
    mv_face_recognition_meta.cpp
)
target_include_directories(gstmvfacerecognizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/worker
    ${CMAKE_SOURCE_DIR}/kernels
    ${CMAKE_SOURCE_DIR}/recognition
    ${GSTREAMER_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDES}
    ${NVINFER_INCLUDE_DIR}
    ${CJSON_INCLUDE_DIRS}
)
target_link_libraries(gstmvfacerecognizer
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_utils
    mergenvision_recognition
    mergenvision_kernels
)
set_target_properties(gstmvfacerecognizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gst-plugins
)
