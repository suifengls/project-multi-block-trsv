cs217\_trsv
==========

multi-block CSR version

performance: analysis - 84 usec, solve - 183 usec

---

trsv.cu:

    1. topology sorting and leveling
    2. set block# and thread# to call GPU\_Kernel
    3. call CPU version of solver

trsv\_kernel.cu:

    multi-block version, currently best we can get.

trsv.h:

    just use for setting size of matrix

trsv\_gold.cpp:

    cpu versions of naive solver and CSR_solver

other\_version:

    contains other trsv_kernel:

    other versions of GPU solvers, performance is not good.

two pdf files:
	
	One is where the idea comes from, one is report on trsv.

---

How to run the program:
--------

1. extract the codes out

2. copy to storm or newton (GPU server)

3. compile program

   -> make

4. run the program

   -> ./trsv
