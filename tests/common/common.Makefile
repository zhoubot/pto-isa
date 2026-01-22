ROOT := $(shell echo $(CURDIR) | sed -e "s@\(.*\)/pto-isa/.*@\1/pto-isa@")
TEST_ROOT := $(shell echo $(CURDIR) | sed -e "s@\(.*\)/pto-isa\/tests/.*@\1/pto-isa\/tests@")
CATEGORY := $(shell echo $(CURDIR) | sed -e 's/\(.*\)pto-isa\/tests\/\(.*\)/\2/')
CATEGORY_NAME := $(shell echo $(CATEGORY) | sed -e 's/\//_/g')
OBJ_ROOT := $(shell realpath $(TEST_ROOT)/../output)
CASE_SRC_DIR := $(CATEGORY)/src
ELF_DIR := $(OBJ_ROOT)/$(CATEGORY)/elf
SRC_DIR := $(shell dirname $(SRC_FILE))
OBJ_DIR := $(OBJ_ROOT)/$(CASE_SRC_DIR)
ELF_HEAD := $(ELF_DIR)/$(CATEGORY_NAME)

OBJ := $(patsubst %.cpp, %.o, $(subst $(SRC_DIR), $(OBJ_DIR), $(SRC_FILE)))
OBJ := $(patsubst %.c, %.o, $(OBJ))
OBJ := $(patsubst %.s, %.o, $(OBJ))
OBJ := $(patsubst %.cce, %.o, $(OBJ))

PLAT ?= dav

ifeq ($(PLAT), dav)
DEFINES += -D__PTO_MANUAL__
#COMPILER_DIR := $(shell echo $(CURDIR) | sed -e "s@\(.*\)/pto-isa/.*@\1/Ascend/latest@")
COMPILER_DIR := /usr/local/Ascend/latest
CCEC_DIR ?= $(COMPILER_DIR)/aarch64-linux/ccec_compiler/bin
#CCEC_DIR ?= $(COMPILER_DIR)/x86_64-linux/ccec_compiler/bin
AS = $(CCEC_DIR)/ccec
CC = $(CCEC_DIR)/ccec 
CXX = $(CCEC_DIR)/ccec
ifeq ($(VERSION), a2a3)
CC_O = -O2 --cce-aicore-arch=dav-c220-vec -lascendcl -lruntime
DEFINES += -DMEMORY_BASE
else ifeq ($(VERSION), a5)
CC_O = -O2 --cce-aicore-arch=dav-c310 -lascendcl -lruntime
DEFINES += -DREGISTER_BASE
endif
CC_VER ?= -std=c++17
INCLUDE += -I$(COMPILER_DIR)/runtime/include -I$(ROOT)/tests/common
INCLUDE += -L$(COMPILER_DIR)/runtime/lib64 -lstdc++ -lgcc

endif

ifeq ($(PY_LIB), on)
CC_O += -fPIC
CC_LINK += -shared
INCLUDE += -I/usr/include/python3.10 -I/usr/local/lib/python3.10/dist-packages/pybind11/include \
           -I$(HOME)/.local/lib/python3.10/site-packages/pybind11/include
endif

ifeq ($(LIB), on)
CC_O += -fPIC
CC_LINK += -shared
endif

INCLUDE += -I$(ROOT)/include -I$(ROOT)/tests/common

CC_O_ALL = $(CC_O) $(CC_VER) $(CC_OPTS)
# $(info ROOT:		$(ROOT))
# $(info CATEGORY:	$(CATEGORY))
# $(info CATEGORY_NAME:	$(CATEGORY_NAME))
# $(info TEST_ROOT:	$(TEST_ROOT))
# $(info SRC_DIR:	$(SRC_DIR))
# $(info SRC_FILE:	$(SRC_FILE))
# $(info OBJ_ROOT:	$(OBJ_ROOT))
# $(info OBJ_DIR:	$(OBJ_DIR))
# $(info OBJ:		$(OBJ))
# $(info ELF_HEAD:	$(ELF_HEAD))
# $(info TARGET:		$(TARGET))

$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	@mkdir -p $(shell dirname $@)
	$(CXX) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.cce
	@mkdir -p $(shell dirname $@)
	$(CXX) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.c
	@mkdir -p $(shell dirname $@)
	$(CC) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

$(OBJ_DIR)%.o: $(SRC_DIR)%.s
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

$(TARGET): $(SRC_FILE)
	mkdir -p $(shell dirname $@)
	$(CC) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $^ -o $@
	chmod 755 $@
	$(info 'You need to set the environment variable before running the binary. Execute the following command line:	\
export LD_LIBRARY_PATH=$(COMPILER_DIR)/runtime/lib64:$$LD_LIBRARY_PATH')


clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;

clean_all:
	@rm -rf $(OBJ_ROOT)
	@find $(TEST_ROOT) -type f -name "*.o" -exec rm -rf {} \;

.PHONY: clean clean_all
