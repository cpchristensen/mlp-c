mnist: mnist-example.c MLP.h MLP.c
	gcc mnist-example.c MLP.c -omnist -std=c89 -pedantic -Wall -Wextra -lm

baseline: baseline-example.c genann/genann.c genann/genann.h
	gcc baseline-example.c genann/genann.c -obaseline -lm
