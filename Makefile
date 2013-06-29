# List of sources, with .c, .cu, and .cc extensions
sources   := trsv.cu trsv_kernel.cu trsv_gold.cpp


# Other things that need to be built, e.g. .cubin files
extradeps := 


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

opt  ?= -O3
dbg  ?= -g
warn ?= -Wall -Werror


# This is where the cuda runtime libraries and includes reside on the
# berkeley millennium/rad clusters. It may be different on your machine.

cudaroot  := /usr/local/cuda
cudaSDKroot := /tmp/NVIDIA_GPU_Computing_SDK/C

#----- C compilation options ------
gcc        := /usr/bin/gcc
cflags     += $(opt) $(dbg) $(warn)
clib_paths :=
cinc_paths := 
clibraries := 


#----- C++ compilation options ------
gpp         := /usr/bin/g++
ccflags     += $(opt) $(dbg) $(warn)
cclib_paths :=
ccinc_paths := -I $(cudaroot)/include
cclibraries := 
# XXLiu: need glut.h for graphics support, in Example of Julia Set.

#----- CUDA compilation options -----

nvcc        := $(cudaroot)/bin/nvcc
cuflags     += $(opt) $(dbg) -arch sm_20
culib_paths := -L$(cudaroot)/lib64 -L$(cudaSDKroot)/lib
cuinc_paths := -I$(cudaroot)/include -I$(cudaSDKroot)/common/inc
culibraries := -lcuda -lcudart 
# -lcublas
lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths)
libraries   := $(culibraries) $(cclibraries) $(clibraries)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c,  $(sources))
ccsources := $(filter %.cc, $(sources)) \
	     $(filter %.cpp,$(sources))
cusources := $(filter %.cu, $(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.c,  %.o,$(csources))  \
           $(patsubst %.cu, %.o,$(cusources)) \
	   $(patsubst %.cpp,%.o,$(filter %.cpp,$(ccsources))) \
	   $(patsubst %.cc, %.o,$(filter %.cc, $(ccsources)))


#----- Build rules ------

# $(target): $(extradeps) 
# 
# 
# $(target): $(objects) 
# 	$(gpp) $(objects) $(lib_paths) $(libraries) -o $@ 

trsv: $(objects)
	$(gpp) trsv_gold.o trsv.o $(lib_paths) $(libraries) -o $@

#----------------------------------
%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(cuinc_paths) -o $@ 

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $^

%.o: %.cc %.cpp
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@

%.o: %.c
	$(gcc) -c $^ $(cflags) $(cinc_paths) -o $@

clean:
	rm -f *.o trsv $(target) makefile.*dep


#----- Dependency Generation -----
#
# If a particular set of sources is non-empty, then have rules for
# generating the necessary dep files.
#

ccdep := ccdep.mk
cdep  := cdep.mk
cudep := cudep.mk


depfiles =

ifneq ($(ccsources),)

depfiles += $(ccdep)
$(ccdep): $(ccsources)
	$(gpp) -MM $(ccsources) > $(ccdep)

else

$(ccdep):

endif

ifneq ($(cusources),)

depfiles += $(cudep)
$(cudep):
	$(gpp) -MM -x c++ $(cusources) > $(cudep)

else

$(cudep):

endif

ifneq ($(csources),)

depfiles += $(cdep)
$(cdep): $(csources)
	$(gcc) -MM -x c $(csources) > $(cdep)

else

$(cdep):

endif

.PHONY: dep
dep: $(depfiles)


# ifneq ($(MAKECMDGOALS),dep)
#  ifneq ($(MAKECMDGOALS),clean)
#   ifneq ($(ccsources),)
#    include $(ccdep)
#   endif
#   ifneq ($(cusources),)
#    include $(cudep)
#   endif
#   ifneq ($(csources),)
#    include $(cdep)
#   endif
#  endif
# endif
