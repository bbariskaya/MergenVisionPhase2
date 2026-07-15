cmake_minimum_required(VERSION 3.20)
project(mergenvision_deepstream_worker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

find_package(CUDAToolkit REQUIRED)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-9.0")
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_ROOT}/sources/includes")
set(DEEPSTREAM_LIBS "${DEEPSTREAM_ROOT}/lib")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${GSTREAMER_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDES}
)

link_directories(
    ${GSTREAMER_LIBRARY_DIRS}
    ${DEEPSTREAM_LIBS}
)

set(CUDA_SOURCES
    kernels/retinaface_decode.cu
    kernels/argsort.cu
    kernels/nms.cu
    kernels/scale_clip_compact_xy.cu
    kernels/l2_normalize.cu
    kernels/similarity_transform.cu
    kernels/warp_align.cu
)

add_library(mergenvision_kernels STATIC ${CUDA_SOURCES})
set_target_properties(mergenvision_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mergenvision_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(mergenvision_kernels CUDA::cudart)

add_library(retinaface_postproc STATIC
    worker/retinaface_postproc.cpp
)
set_target_properties(retinaface_postproc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(retinaface_postproc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(retinaface_postproc mergenvision_kernels CUDA::cudart)

add_library(retinaface_parser SHARED
    parsers/retinaface_parser.cpp
)
target_include_directories(retinaface_parser PRIVATE ${DEEPSTREAM_INCLUDES})

add_executable(deepstream_face_worker
    worker/main.cpp
)

target_link_libraries(deepstream_face_worker
    retinaface_postproc
    mergenvision_kernels
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_inferutils
    nvds_utils
    nvds_infercustomparser
)

add_subdirectory(plugins/gst-nvdsretinaface)

add_executable(render_annotated_video
    tools/render_annotated_video.cpp
)
target_link_libraries(render_annotated_video
    retinaface_postproc
    mergenvision_kernels
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_inferutils
    nvds_utils
    nvds_infercustomparser
)

add_executable(test_nms
    tests/test_nms.cu
)
target_link_libraries(test_nms
    mergenvision_kernels
    CUDA::cudart
)

# Recognition executable can be added later.
