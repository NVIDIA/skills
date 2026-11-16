# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

FTN    := gfortran
FFLAGS := -O2

EXEC := advection_code

.PHONY: all run clean

all: $(EXEC)

$(EXEC): advection_code.f90
	$(FTN) $(FFLAGS) $< -o $@

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f *.o *.mod $(EXEC)
