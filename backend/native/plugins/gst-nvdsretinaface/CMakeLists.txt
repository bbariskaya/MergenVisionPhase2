find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0)

find_package(CUDAToolkit REQUIRED)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-9.0")
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_ROOT}/sources/includes")
set(PREPROCESS_INCLUDE "${DEEPSTREAM_ROOT}/sources/gst-plugins/gst-nvdspreprocess/include")

add_library(retinaface_engine STATIC
    retinaface_engine.cpp
)
set_target_properties(retinaface_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(retinaface_engine PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDES}
)
target_link_libraries(retinaface_engine
    nvinfer
    CUDA::cudart
)

add_library(gstnvdsretinaface SHARED
    gstnvdsretinaface.cpp
)
target_include_directories(gstnvdsretinaface PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/worker
    ${CMAKE_SOURCE_DIR}/kernels
    ${GSTREAMER_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDES}
    ${PREPROCESS_INCLUDE}
)
target_link_libraries(gstnvdsretinaface
    retinaface_engine
    retinaface_postproc
    mergenvision_kernels
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_utils
)
set_target_properties(gstnvdsretinaface PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Keep our plugin in a dedicated directory so the whole build tree is not
    # added to GST_PLUGIN_PATH.
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gst-plugins
)
