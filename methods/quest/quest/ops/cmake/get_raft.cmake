# quest/ops/cmake/get_raft.cmake



set(RAFT_VERSION "24.02")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAFT_VERSION}")

function(find_and_configure_raft)

  set(oneValueArgs VERSION FORK PINNED_TAG COMPILE_LIBRARIES)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  find_package(raft CONFIG QUIET)
  if(raft_FOUND AND TARGET raft::raft)
    message(STATUS "Using installed RAFT package")
    return()
  endif()


  rapids_cpm_find(raft ${PKG_VERSION}
    GLOBAL_TARGETS raft::raft
    CPM_ARGS
      GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
      GIT_TAG        ${PKG_PINNED_TAG}
      SOURCE_SUBDIR  cpp
      OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCH OFF"
        "RAFT_COMPILE_LIBRARIES ${PKG_COMPILE_LIBRARIES}"
  )
endfunction()



#   -DCPM_raft_SOURCE=/path/to/raft/cpp


find_and_configure_raft(
  VERSION           ${RAFT_VERSION}.00   # → 24.02.00
  FORK              ${RAFT_FORK}
  PINNED_TAG        ${RAFT_PINNED_TAG}   # → branch-24.02
  COMPILE_LIBRARIES NO
)
