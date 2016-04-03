Make sure you compile the code with the CUDA compiler and have the openmp flags set to be able to use the different versions. To enable the openmp flag for this project you have to edit the nvcc.profile file located in the bin folder of your CUDA toolkit installation<sup>1</sup>.

Open nvcc.profile as Administrator and change

INCLUDES        +=  “-I$(TOP)/include” $(_SPACE_)

to

INCLUDES        +=  “-I$(TOP)/include” “/openmp” $(_SPACE_)

There is no graphical menu, to find out about the different options go into the code and there you can find the different keys that have actions attached.

The most important bit is the fact that 0, 1 and 2 change the place where the calculations are done {CPU, OpenMP and CUDA} respectively.

Enjoy!

[1] Thanks to http://www.orangeowlsolutions.com/archives/783
