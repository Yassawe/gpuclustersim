set(LIBRARY_NAME gcs_bindings)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} MODULE ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_python_library gcs_sim_engine gcs_torch_backend)

target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${LIBRARY_NAME} PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
)

if(WIN32)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${Python3_LIBRARIES})
elseif(APPLE)
    set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()



install(TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)