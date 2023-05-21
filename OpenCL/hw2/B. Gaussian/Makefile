IDIR="C:/Program Files (x86)/OCL_SDK_Light/include/"
CC=gcc
CFLAGS=-I $(IDIR) -L $(LDIR)

ODIR="./bin/"
LDIR="C:/Program Files (x86)/OCL_SDK_Light/lib/x86"

LIBS=-lm -fopenmp -lOpenCL

DEPS = cl.h

OBJ1 = array-addition.o 
OBJ2 = gaussian-blur.o 

$(ODIR)%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

array-addition: $(OBJ1)
	$(CC) -o $(ODIR)$@ $^ $(CFLAGS) $(LIBS)

gaussian-blur: $(OBJ2)
	$(CC) -o $(ODIR)$@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 

