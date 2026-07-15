cmake_minimum_required(VERSION 3.20)
project(mergenvision_deepstream_worker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(CJSON REQUIRED libcjson)

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)

find_library(NVINFER_LIBRARY nvinfer
    PATHS /usr/lib/x86_64-linux-gnu)
find_path(NVINFER_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include)
if(NOT NVINFER_LIBRARY OR NOT NVINFER_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT (nvinfer) not found")
endif()

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
    kernels/warp_align_rgba_pitch.cu
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

add_library(mergenvision_recognition STATIC
    recognition/glintr100_engine.cpp
    recognition/gallery.cpp
)
set_target_properties(mergenvision_recognition PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mergenvision_recognition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/recognition
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVINFER_INCLUDE_DIR}
    ${CJSON_INCLUDE_DIRS}
)
target_link_libraries(mergenvision_recognition
    CUDA::cudart
    ${NVINFER_LIBRARY}
    ${CJSON_LIBRARIES}
    OpenSSL::Crypto
)

add_library(retinaface_parser SHARED
    parsers/retinaface_parser.cpp
)
target_include_directories(retinaface_parser PRIVATE ${DEEPSTREAM_INCLUDES})

add_executable(deepstream_face_worker
    worker/main.cpp
    plugins/gst-mvfacerecognizer/mv_face_recognition_meta.cpp
)

target_include_directories(deepstream_face_worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/recognition
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins/gst-mvfacerecognizer
)
target_link_libraries(deepstream_face_worker
    retinaface_postproc
    mergenvision_kernels
    mergenvision_recognition
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_inferutils
    nvds_utils
    nvds_infercustomparser
)

add_subdirectory(tracking)
add_subdirectory(plugins/gst-nvdsretinaface)
add_subdirectory(plugins/gst-mvfacerecognizer)
add_subdirectory(plugins/gst-mvfacetracker)

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

add_executable(second_pass_render
    tools/second_pass_render.cpp
)
target_link_libraries(second_pass_render
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_utils
)

add_executable(sprint04_feasibility_probe
    tools/sprint04_feasibility_probe.cpp
)
target_link_libraries(sprint04_feasibility_probe
    ${GSTREAMER_LIBRARIES}
    CUDA::cudart
    nvds_meta
    nvdsgst_meta
    nvds_inferutils
    nvds_utils
)

add_executable(test_nms
    tests/test_nms.cu
)
target_link_libraries(test_nms
    mergenvision_kernels
    CUDA::cudart
)

add_executable(test_warp_align_rgba_pitch
    tests/test_warp_align_rgba_pitch.cu
)
target_link_libraries(test_warp_align_rgba_pitch
    mergenvision_kernels
    CUDA::cudart
)

# Recognition executable can be added later.

add_executable(test_l2_normalize
    tests/test_l2_normalize.cu
)
target_link_libraries(test_l2_normalize
    mergenvision_kernels
    CUDA::cudart
)

add_executable(test_gallery
    tests/test_gallery.cpp
)
target_include_directories(test_gallery PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/recognition
    ${CJSON_INCLUDE_DIRS}
)
target_link_libraries(test_gallery
    mergenvision_recognition
)

add_executable(test_glintr100_engine
    tests/test_glintr100_engine.cpp
)
target_include_directories(test_glintr100_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/recognition
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVINFER_INCLUDE_DIR}
)
target_link_libraries(test_glintr100_engine
    mergenvision_recognition
    CUDA::cudart
)
