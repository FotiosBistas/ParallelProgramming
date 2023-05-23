CC=gcc

ODIR="./bin/"
IMAGEDIR = "./producedImages/"

LIBS=-lm -fopenmp 



gaussian-blur: gaussian-blur.c
	$(CC) -o $(ODIR)$@ $^ $(LIBS) 


clear-images: 
	rm -f $(IMAGEDIR)*.bmp  *~ core $(INCDIR)/*~

.PHONY: clean

clean:
	rm -f $(ODIR)* *~ core $(INCDIR)/*~ 

