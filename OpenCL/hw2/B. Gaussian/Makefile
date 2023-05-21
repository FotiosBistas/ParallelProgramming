IDIR="C:/Program Files (x86)/OCL_SDK_Light/include/"
CC=gcc
CFLAGS=-I $(IDIR) -L $(LDIR)

ODIR="./bin/"
LDIR="C:/Program Files (x86)/OCL_SDK_Light/lib/x86"

LIBS=-lm -lOpenCL

DEPS = cl.h

OBJ = array-addition.o 


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

array-addition: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 

