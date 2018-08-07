# For my MacBook
#FFTW = -L ~/cmb/FFTW/lib -I ~/cmb/FFTW/include -lfftw3
#CXXFLAGS = -Wall -O2
#CXXFLAGS = -g ${FFTW}

# For Odyssey
#FFTW = -L ~/fftw-3.3.5/lib/ -I ~/fftw-3.3.5/include/ -lfftw3 # for Daniel
FFTW = -L ~/_fftw-3.3.5/api/ -I ~/_fftw-3.3.5/api/ -lfftw3   # for Ryuichiro  Do I need this????

OMP = -DOPENMP -DFFTSLAB -DSLAB

# If you want to run with multi-threading, uncomment the following two lines
CXX = g++ -march=native -fopenmp -lgomp -fopt-info-vec-missed -fopt-info-vec-optimized
CXXFLAGS = -O3 ${OMP}

# Or is you want multi-threading with icc, the following would work:
# ICC not tested!  And one may need to compile FFTW with it.
#CXX = icc -liomp5 -openmp
#CXXFLAGS = -O2 -Wall ${OMP}

default: reconst

reconst: reconst.cpp Makefile merge_sort_omp.cpp STimer.cc
	${CXX} ${CXXFLAGS} reconst.cpp ${FFTW} -o reconst

clean:
	rm reconst
	
tar:
	tar cvf fftcorr.tar --exclude="*.pyc" \
      Makefile STimer.cc fftcorr.cpp fftcorr.py wcdm merge_sort_omp.cpp d12.cpp generate_command.py
