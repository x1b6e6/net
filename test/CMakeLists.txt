add_custom_target(tests)

function (new_test name)
	add_executable(test_${name} EXCLUDE_FROM_ALL ${name}.cc)
	set_target_properties(test_${name} PROPERTIES 
		CXX_STANDARD 20 
		CXX_STANDARD_REQUIRED ON)
	target_link_libraries(test_${name} net::net)
	add_test(${name} test_${name})
	add_dependencies(tests test_${name})
endfunction()

new_test(simple_xor)
new_test(net_xor)

# vim: set ts=4 sw=4 :
