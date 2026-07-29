set(LIBRARY_NAME gcs_bindings)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Module.cpp
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python_library gcs_torch_backend)


if(WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${Python3_LIBRARIES})
elseif(APPLE)
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)