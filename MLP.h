#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
	float** layers;
	float** activation_derivatives;
	float*** weights;

	int hidden_layers_n;
	int nodes_per_layer;
	int input_nodes_n;
	int output_nodes_n;
	int memory_usage;

	float learning_rate;
} ANN;

ANN* initANN(int input_nodes_n, int hidden_layers_n, int nodes_per_layer, int output_nodes_n, float learning_rate);

void randomizeANNWeights (ANN* ann);
float activationANN (float a);

float* runANN (ANN* ann, float* input_values);
float trainANN (ANN* ann, float* input_values, float* expected);

void saveANN (ANN* ann, const char* filename);
ANN* loadANN (const char* filename);

ANN* copyANN (ANN* ann);
void mutateANN (ANN* ann, float mutation_rate, float mutation_magnitude);

void freeANN(ANN* ann);

#endif
