# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /usr/share/doc/ultraleap-hand-tracking-service/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example

# Include any dependencies generated for this target.
include CMakeFiles/libExampleConnection.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/libExampleConnection.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/libExampleConnection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libExampleConnection.dir/flags.make

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o: CMakeFiles/libExampleConnection.dir/flags.make
CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o: /usr/share/doc/ultraleap-hand-tracking-service/samples/ExampleConnection.c
CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o: CMakeFiles/libExampleConnection.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o -MF CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o.d -o CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o -c /usr/share/doc/ultraleap-hand-tracking-service/samples/ExampleConnection.c

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /usr/share/doc/ultraleap-hand-tracking-service/samples/ExampleConnection.c > CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /usr/share/doc/ultraleap-hand-tracking-service/samples/ExampleConnection.c -o CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s

libExampleConnection: CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o
libExampleConnection: CMakeFiles/libExampleConnection.dir/build.make
.PHONY : libExampleConnection

# Rule to build all files generated by this target.
CMakeFiles/libExampleConnection.dir/build: libExampleConnection
.PHONY : CMakeFiles/libExampleConnection.dir/build

CMakeFiles/libExampleConnection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libExampleConnection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libExampleConnection.dir/clean

CMakeFiles/libExampleConnection.dir/depend:
	cd /home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /usr/share/doc/ultraleap-hand-tracking-service/samples /usr/share/doc/ultraleap-hand-tracking-service/samples /home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example /home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example /home/david/ultraleap-tracking-samples/build/Release/LeapSDK/leapc_example/CMakeFiles/libExampleConnection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libExampleConnection.dir/depend
