# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aniruddha/Downloads/libnabo-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /usr/local/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/knnepsilon.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/knnepsilon.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/knnepsilon.dir/flags.make

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o: tests/CMakeFiles/knnepsilon.dir/flags.make
tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o: /home/aniruddha/Downloads/libnabo-master/tests/knnepsilon.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o"
	cd /usr/local/build/tests && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o -c /home/aniruddha/Downloads/libnabo-master/tests/knnepsilon.cpp

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/knnepsilon.dir/knnepsilon.cpp.i"
	cd /usr/local/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aniruddha/Downloads/libnabo-master/tests/knnepsilon.cpp > CMakeFiles/knnepsilon.dir/knnepsilon.cpp.i

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/knnepsilon.dir/knnepsilon.cpp.s"
	cd /usr/local/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aniruddha/Downloads/libnabo-master/tests/knnepsilon.cpp -o CMakeFiles/knnepsilon.dir/knnepsilon.cpp.s

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.requires:

.PHONY : tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.requires

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.provides: tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.requires
	$(MAKE) -f tests/CMakeFiles/knnepsilon.dir/build.make tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.provides.build
.PHONY : tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.provides

tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.provides.build: tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o


# Object files for target knnepsilon
knnepsilon_OBJECTS = \
"CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o"

# External object files for target knnepsilon
knnepsilon_EXTERNAL_OBJECTS =

tests/knnepsilon: tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o
tests/knnepsilon: tests/CMakeFiles/knnepsilon.dir/build.make
tests/knnepsilon: libnabo.a
tests/knnepsilon: tests/CMakeFiles/knnepsilon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/usr/local/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable knnepsilon"
	cd /usr/local/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/knnepsilon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/knnepsilon.dir/build: tests/knnepsilon

.PHONY : tests/CMakeFiles/knnepsilon.dir/build

tests/CMakeFiles/knnepsilon.dir/requires: tests/CMakeFiles/knnepsilon.dir/knnepsilon.cpp.o.requires

.PHONY : tests/CMakeFiles/knnepsilon.dir/requires

tests/CMakeFiles/knnepsilon.dir/clean:
	cd /usr/local/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/knnepsilon.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/knnepsilon.dir/clean

tests/CMakeFiles/knnepsilon.dir/depend:
	cd /usr/local/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aniruddha/Downloads/libnabo-master /home/aniruddha/Downloads/libnabo-master/tests /usr/local/build /usr/local/build/tests /usr/local/build/tests/CMakeFiles/knnepsilon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/knnepsilon.dir/depend

