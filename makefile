CC      = nvcc
CU_FILES = $(wildcard *.cu)
HEADER= $(wildcard *.h)
OBJS    = $(patsubst %.cu,%.o,$(CU_FILES))
CUFLAGS  =
TARGET  = project

all: $(TARGET)
$(TARGET): $(OBJS) 
	$(CC) $(OBJS) -o $(TARGET)
	
%.o: %.cu $(HEADER)
	$(CC) $(CUFLAGS) -c -o $@ $<
clean:
	rm -f $(OBJS) $(TARGET)

