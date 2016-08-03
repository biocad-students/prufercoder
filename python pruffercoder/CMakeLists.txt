cmake_minimum_required(VERSION 2.8.3)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
#FIND_PACKAGE(RDKit)
FIND_PACKAGE(Boost COMPONENTS python)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} /usr/local/Cellar/rdkit/2016.03.3/include/rdkit/
)
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} 
/usr/local/Cellar/rdkit/2016.03.3/lib/libGraphMol.2016.03.3.dylib 
/usr/local/Cellar/rdkit/2016.03.3/lib/libSmilesParse.2016.03.3.dylib
/usr/local/Cellar/rdkit/2016.03.3/lib/libSubstructMatch.2016.03.3.dylib)

add_subdirectory (atomCode)
add_subdirectory (pruffer) 
add_subdirectory (main) 