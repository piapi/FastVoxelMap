project(proto_msg)

message("${Magenta}-------------- build ${PROJECT_NAME} begin ----------${ColorReset}")

if (WIN32)
  find_package(Protobuf REQUIRED)
else ()
  find_package(Protobuf 3.6.1 REQUIRED)
endif ()

message(STATUS "Protobuf_VERSION: " ${Protobuf_VERSION})
message(STATUS "Protobuf_INCLUDE_DIR: " ${Protobuf_INCLUDE_DIR})
message(STATUS "Protobuf_LIBRARY_RELEASE: " ${Protobuf_LIBRARY_RELEASE})
#message(STATUS "Protobuf_LITE_LIBRARY_RELEASE: " ${Protobuf_LITE_LIBRARY_RELEASE})
#message(STATUS "Protobuf_PROTOC_LIBRARY_RELEASE: " ${Protobuf_PROTOC_LIBRARY_RELEASE})
#message(STATUS "Protobuf_PROTOC_EXECUTABLE: " ${Protobuf_PROTOC_EXECUTABLE})


# 生成中间文件的路径(.cc .h)
set(PROTO_META_BASE_DIR ${CMAKE_BINARY_DIR}/message)
if (EXISTS "${CMAKE_BINARY_DIR}/message" AND IS_DIRECTORY "${CMAKE_BINARY_DIR}/message")
  message(STATUS "${PROTO_META_BASE_DIR} check ok!")
else ()
  file(MAKE_DIRECTORY ${PROTO_META_BASE_DIR})
endif ()

message(STATUS "proto_msg meta files path: ${PROTO_META_BASE_DIR}")

# PROTO_FLAGS指定protoc编译.proto文件时的总的搜索路径，.proto中的import命令根据根据这个地址去连接其他的.proto文件
list(APPEND PROTO_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR})
# 递归查找需要编译的proto文件
file(GLOB_RECURSE MSG_PROTOS ${CMAKE_CURRENT_SOURCE_DIR}/*.proto)
#message(STATUS "MSG_PROTOS: ${MSG_PROTOS}")

set(MESSAGE_HDRS "")
set(MESSAGE_SRC "")

# 逐个文件配置生成命令
foreach (msg ${MSG_PROTOS})
  # 从文件路径中获取文件名
  get_filename_component(FIL_WE ${msg} NAME_WE)

  # 生成编译所需的所有依赖文件
  list(APPEND MESSAGE_SRC "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.cc")
  list(APPEND MESSAGE_HDRS "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.h")

  # message(STATUS "generating ${FIL_WE}.pb.cc & ${FIL_WE}.pb.h")

  # 直接执行命令(不建议直接生成的方法，此方法会每次加载CMakeLists时生成消息中间文件)
  #    execute_process(
  #            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_META_BASE_DIR} ${msg}
  #    )

  if (NOT MSVC)

    # 使用自定义命令，避免编译时每次都要重新编译proto
    add_custom_command(
        OUTPUT "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.cc"
        "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.h"
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_META_BASE_DIR} ${msg}
        DEPENDS ${msg}
        COMMENT "Running C++ protocol buffer compiler on ${msg}"
        VERBATIM
    )
  else ()
    # msvc下生成方法  --cpp_out=dllexport_decl=Protobuf_EXPORT:${PROTO_META_BASE_DIR}
    add_custom_command(
        OUTPUT "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.cc"
        "${PROTO_META_BASE_DIR}/${FIL_WE}.pb.h"
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_META_BASE_DIR} ${msg}
        DEPENDS ${msg}
        COMMENT "Running C++ protocol buffer compiler on ${msg}"
        VERBATIM
    )
  endif ()
endforeach ()

# 设置文件属性为GENERATED
set_source_files_properties(${MESSAGE_SRC} ${MESSAGE_HDRS} PROPERTIES GENERATED TRUE)

# 添加自定义目标，与add_custom_command配合使用
add_custom_target(generate_message ALL
    DEPENDS ${MESSAGE_SRC} ${MESSAGE_HDRS}
    COMMENT "generate message target"
    VERBATIM
    )

if (NOT MSVC)
  # 将所有的msg生成动态库（与其他代码完整隔离）
  add_library(${PROJECT_NAME} SHARED
      ${MESSAGE_SRC}
      )
  # 设置指定的生成路径
  # set_target_properties(${PROJECT_NAME} PROPERTIES
  #     LIBRARY_OUTPUT_DIRECTORY ${SHARED_LIB_PATH}
  #     )
else ()
  # 将所有的msg生成静态库
  add_library(${PROJECT_NAME} STATIC
      ${MESSAGE_SRC}
      )
  #  target_compile_definitions(${PROJECT_NAME}
  #      PRIVATE
  #      RoProtobuf_LIBRARY
  #      )
  target_compile_options(${PROJECT_NAME}
      PRIVATE
      /wd4267 # 从“size_t”转换到“uint32_t”，可能丢失数据
      PUBLIC
      /wd4251 # protobuf warning
      )
endif ()

# 加入编译时所需头文件路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${PROTO_META_BASE_DIR}
    ${Protobuf_INCLUDE_DIR}
    )

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${Protobuf_LIBRARY_RELEASE}
    # ${Protobuf_LITE_LIBRARY_RELEASE}
    # ${Protobuf_PROTOC_LIBRARY_RELEASE}
    )

set_target_properties(${PROJECT_NAME}
    PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

message("${Magenta}-------------- build ${PROJECT_NAME} end ----------${ColorReset}")
