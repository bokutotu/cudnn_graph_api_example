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
CMAKE_SOURCE_DIR = /home/bokutotu/HDD/CUDA/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bokutotu/HDD/CUDA/code/b

# Include any dependencies generated for this target.
include CMakeFiles/graph_api_batch_norm_2d.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/graph_api_batch_norm_2d.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/graph_api_batch_norm_2d.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/graph_api_batch_norm_2d.dir/flags.make

CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o: CMakeFiles/graph_api_batch_norm_2d.dir/flags.make
CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o: ../graph_api_batch_norm_2d.c
CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o: CMakeFiles/graph_api_batch_norm_2d.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bokutotu/HDD/CUDA/code/b/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o -MF CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o.d -o CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o -c /home/bokutotu/HDD/CUDA/code/graph_api_batch_norm_2d.c

CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/bokutotu/HDD/CUDA/code/graph_api_batch_norm_2d.c > CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.i

CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/bokutotu/HDD/CUDA/code/graph_api_batch_norm_2d.c -o CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.s

# Object files for target graph_api_batch_norm_2d
graph_api_batch_norm_2d_OBJECTS = \
"CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o"

# External object files for target graph_api_batch_norm_2d
graph_api_batch_norm_2d_EXTERNAL_OBJECTS =

graph_api_batch_norm_2d: CMakeFiles/graph_api_batch_norm_2d.dir/graph_api_batch_norm_2d.c.o
graph_api_batch_norm_2d: CMakeFiles/graph_api_batch_norm_2d.dir/build.make
graph_api_batch_norm_2d: libutils.a
graph_api_batch_norm_2d: /usr/local/cuda/lib64/libcudart_static.a
graph_api_batch_norm_2d: /usr/lib/x86_64-linux-gnu/librt.a
graph_api_batch_norm_2d: /usr/lib/x86_64-linux-gnu/libcudnn.so
graph_api_batch_norm_2d: CMakeFiles/graph_api_batch_norm_2d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bokutotu/HDD/CUDA/code/b/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable graph_api_batch_norm_2d"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph_api_batch_norm_2d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/graph_api_batch_norm_2d.dir/build: graph_api_batch_norm_2d
.PHONY : CMakeFiles/graph_api_batch_norm_2d.dir/build

CMakeFiles/graph_api_batch_norm_2d.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/graph_api_batch_norm_2d.dir/cmake_clean.cmake
.PHONY : CMakeFiles/graph_api_batch_norm_2d.dir/clean

CMakeFiles/graph_api_batch_norm_2d.dir/depend:
	cd /home/bokutotu/HDD/CUDA/code/b && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bokutotu/HDD/CUDA/code /home/bokutotu/HDD/CUDA/code /home/bokutotu/HDD/CUDA/code/b /home/bokutotu/HDD/CUDA/code/b /home/bokutotu/HDD/CUDA/code/b/CMakeFiles/graph_api_batch_norm_2d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/graph_api_batch_norm_2d.dir/depend

