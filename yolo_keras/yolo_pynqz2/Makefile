YOLO = yolo

CXX       :=   g++
CC        :=   gcc 
YOLOOBJ :=   runYolo.o

# linking libraries of OpenCV
LDFLAGS  =  $(shell pkg-config --libs opencv) -lpthread -lhineon -ln2cube -ldputils

CUR_DIR =   $(shell pwd)
MODEL   =   $(CUR_DIR)/dpu_testModel.elf
ARCH    =     $(shell uname -m | sed -e s/arm.*/armv71/ \
	-e s/aarch64.*/aarch64/ )

CFLAGS  :=  -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math

ifeq ($(ARCH),armv71)
	CFLAGS +=  -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon
endif
ifeq ($(ARCH),aarch64)
	CFLAGS += -mcpu=cortex-a53
endif

.PHONY: clean
%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(CUR_DIR)/$@


$(YOLO) : $(YOLOOBJ)
	$(CXX) $(CFLAGS) $(addprefix $(CUR_DIR)/, $^) $(MODEL) -o $(CUR_DIR)/$@ $(LDFLAGS)


