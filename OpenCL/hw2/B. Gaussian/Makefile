IDIR="C:/Program Files (x86)/OCL_SDK_Light/include/"
CC=gcc
CFLAGS=-I $(IDIR) -L $(LDIR) 

ODIR="./bin/"
LDIR="C:/Program Files (x86)/OCL_SDK_Light/lib/x86"
IMAGEDIR = "./producedImages/"

LIBS=-lm -fopenmp -lOpenCL


array-addition: array-addition.c 
	$(CC) -o $(ODIR)$@ $^ $(CFLAGS) $(LIBS)

gaussian-blur: gaussian-blur.c
	$(CC) -o $(ODIR)$@ $^ $(CFLAGS)  $(LIBS) 


clear-images: 
	rm -f $(IMAGEDIR)*.bmp  *~ core $(INCDIR)/*~

.PHONY: clean

clean:
	rm -f *.o *~ core $(INCDIR)/*~ 

