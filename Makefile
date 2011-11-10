# set EXTRA_CMAKE_FLAGS in the including Makefile in order to add tweaks
CMAKE_FLAGS= -Wdev $(EXTRA_CMAKE_FLAGS)

# The all target does the heavy lifting, creating the build directory and
# invoking CMake
all:
	@mkdir -p build
	-mkdir -p bin
	cd build && cmake $(CMAKE_FLAGS) ..
	cd build && make 

clean:
	-cd build && make clean
	rm -rf build

eclipse-project: 
	mv Makefile Makefile.bla
	cmake -G"Eclipse CDT4 - Unix Makefiles" -Wno-dev .
	rm Makefile
	rm CMakeCache.txt
	rm -rf CMakeFiles
	mv Makefile.bla Makefile
