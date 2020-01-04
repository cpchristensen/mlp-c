#include "MLP.h"

ANN* initANN (int input_nodes_n, int hidden_layers_n, int nodes_per_layer, int output_nodes_n, float learning_rate) {
	int i, j;

	ANN* ann = malloc(sizeof(ANN));
	ann->memory_usage = sizeof(ANN);

	ann->hidden_layers_n = hidden_layers_n;
	ann->nodes_per_layer = nodes_per_layer;
	ann->input_nodes_n = input_nodes_n;
	ann->output_nodes_n = output_nodes_n;
	ann->learning_rate = learning_rate;

	/* Total number of layers. */
	ann->layers = malloc((hidden_layers_n + 2) * sizeof(float*));
	ann->memory_usage += (hidden_layers_n + 2) * sizeof(float*);

	ann->activation_derivatives = malloc((hidden_layers_n + 2) * sizeof(float*));
	ann->memory_usage += (hidden_layers_n + 2) * sizeof(float*);

	/* Input layer */
	ann->layers[0] = malloc(input_nodes_n * sizeof(float));
	ann->memory_usage += input_nodes_n * sizeof(float);

	ann->activation_derivatives[0] = malloc(input_nodes_n * sizeof(float));
	ann->memory_usage += input_nodes_n * sizeof(float);

	/* Output layer. */
	ann->layers[hidden_layers_n + 1] = malloc(output_nodes_n * sizeof(float));
	ann->memory_usage += output_nodes_n * sizeof(float);

	ann->activation_derivatives[hidden_layers_n + 1] = malloc(output_nodes_n * sizeof(float));
	ann->memory_usage += output_nodes_n * sizeof(float);

	/* Hidden layers. */
	for (i = 1; i < (hidden_layers_n + 1); i++) {
		ann->layers[i] = malloc(nodes_per_layer * sizeof(float));
		ann->memory_usage += nodes_per_layer * sizeof(float);

		ann->activation_derivatives[i] = malloc(nodes_per_layer * sizeof(float));
		ann->memory_usage += nodes_per_layer * sizeof(float);
	}

	ann->weights = malloc((hidden_layers_n + 1) * sizeof(float**));
	ann->memory_usage += (hidden_layers_n + 1) * sizeof(float**);

	for (i = 0; i < (hidden_layers_n + 1); i++) {
		ann->weights[i] = malloc( ((i == 0 ? input_nodes_n : nodes_per_layer) + 1) * sizeof(float*));
		ann->memory_usage += ((i == 0 ? input_nodes_n : nodes_per_layer) + 1) * sizeof(float*);

		for (j = 0; j < ((i == 0 ? input_nodes_n : nodes_per_layer) + 1); j++) {
			int new_size = ((i + 1) == (hidden_layers_n + 1)) ? output_nodes_n : nodes_per_layer;
			ann->weights[i][j] = malloc(new_size * sizeof(float));
			ann->memory_usage += new_size * sizeof(float);
		}
	}

	randomizeANNWeights(ann);

	return ann;
}

void randomizeANNWeights (ANN* ann) {
	int i, j, k;
	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			for (k = 0; k < ((i + 1) == (ann->hidden_layers_n + 1) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
				ann->weights[i][j][k] = ((float) rand() / (float) RAND_MAX) - 0.5;
			}
		}
	}
}

float activationANN (float a) {
	/*
	 * Does not bother with calculation if value is too small or too large.
	 * This simulates the "approaching 1" and "approaching 0" properties.
	 */
	if (a > 30.0) return 1.0;
	else if (a < -30.0) return 0.0;


	return pow(1.0 + exp(-a), -1.0);
}
float* runANN (ANN* ann, float* input_values) {
	int i, j, k;
	float sum;

	for (i = 0; i < ann->input_nodes_n; i++) {
		ann->layers[0][i] = input_values[i];
	}

	for (i = 1; i < (ann->hidden_layers_n + 2); i++) {
		for (j = 0; j < ((i == (ann->hidden_layers_n + 1)) ? ann->output_nodes_n : ann->nodes_per_layer); j++) {
			sum = 0.0;

			for (k = 0; k < (((i - 1) == 0) ? ann->input_nodes_n : ann->nodes_per_layer); k++) {
				sum += (ann->layers[i - 1][k] * ann->weights[i - 1][k][j]);

			}

			sum += ann->weights[i - 1][(((i - 1) == 0) ? ann->input_nodes_n : ann->nodes_per_layer) - 1][j];

			ann->layers[i][j] = activationANN(sum);

		}
	}

	return ann->layers[ann->hidden_layers_n + 1];
}

float trainANN (ANN* ann, float* input_values, float* expected) {
	int i, j, k;
	/* Calculates the cost of the current dataset. */
	float cost = 0.0;

	float act_d;
	float z_d;
	float previous_activation;

	int current_size = 0;

	/* First feeds forward, so cost can be calculated. */
	runANN(ann, input_values);

	for (i = 0; i < ann->output_nodes_n; i++) {
		cost += pow(ann->layers[ann->hidden_layers_n + 1][i] - expected[i], 2.0);
	}

	/* Stores the derivative of the cost with the current activation node. */
	for (i = 0; i < (ann->hidden_layers_n + 2); i++) {
		if (i == 0) {
			current_size = ann->input_nodes_n;
		} else if (i == (ann->hidden_layers_n + 1)) {
			current_size = ann->output_nodes_n;
		} else {
			current_size = ann->nodes_per_layer;
		}

		for (j = 0; j < current_size; j++) {
			ann->activation_derivatives[i][j] = ann->layers[i][j];
		}
	}

	/* Backpropogates through the network. */
	for (i = (ann->hidden_layers_n + 1); i > 0; i--) {
		for (j = 0; j < ((i == (ann->hidden_layers_n + 1)) ? ann->output_nodes_n : ann->nodes_per_layer); j++) {
			act_d = 0.0;
			/* The first activation derivative cMLPot be defined by the previous derivatives. */
			if (i == (ann->hidden_layers_n + 1)) {
				act_d = 2.0 * (ann->layers[i][j] - expected[j]);
			/* The following activation derivatives are all defined by previous ones. */
			} else {
				for (k = 0; k < (((i + 1) == (ann->hidden_layers_n + 1)) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
					act_d += ann->weights[i][j][k] * ann->activation_derivatives[i + 1][k];
				}
			}

			/* Derivative of the activation function. */
			z_d = ann->layers[i][j] * (1.0 - ann->layers[i][j]);
			/* Sets up activation derivative for next iteration. */
			ann->activation_derivatives[i][j] = act_d * z_d;

			/* Code that actually updates the weights and biases. */
			for (k = 0; k < ((i - 1) == 0 ? ann->input_nodes_n : ann->nodes_per_layer); k++) {
				previous_activation = ann->layers[i - 1][k];
				/* Updates weights. */
				ann->weights[i - 1][k][j] -= (ann->activation_derivatives[i][j] * previous_activation * ann->learning_rate);
			}
			/* Updates biases. */
			ann->weights[i - 1][((i - 1) == 0 ? ann->input_nodes_n : ann->nodes_per_layer) - 1][j] -= (act_d * z_d * ann->learning_rate);
		}
	}

	return cost;
}

void saveANN (ANN* ann, const char* filename) {
	int i, j, k;
	FILE* fp = fopen(filename, "w");

	fprintf(fp, "%d\n%d\n%d\n%d\n%f\n", ann->input_nodes_n, ann->hidden_layers_n, ann->nodes_per_layer, ann->output_nodes_n, ann->learning_rate);

	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			for (k = 0; k < ((i + 1) == (ann->hidden_layers_n + 1) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
				fprintf(fp, "%f\n", ann->weights[i][j][k]);
			}
		}
	}

	fclose(fp);

	return;
}

ANN* loadANN (const char* filename) {
	int i, j, k;
	int input_nodes_n, hidden_layers_n, nodes_per_layer, output_nodes_n;
	float learning_rate;

	ANN* ann;

	FILE* fp = fopen(filename, "r");
	if (!fp) {
		return NULL;
	}

	fscanf(fp, "%d\n%d\n%d\n%d\n%f\n", &input_nodes_n, &hidden_layers_n, &nodes_per_layer, &output_nodes_n, &learning_rate);

	ann = initANN(input_nodes_n, hidden_layers_n, nodes_per_layer, output_nodes_n, learning_rate);

	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			for (k = 0; k < ((i + 1) == (ann->hidden_layers_n + 1) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
				fscanf(fp, "%f\n", &ann->weights[i][j][k]);
			}
		}
	}

	fclose(fp);

	return ann;
}

ANN* copyANN (ANN* ann) {
	int i, j, k;

	ANN* cpy = initANN(ann->input_nodes_n, ann->hidden_layers_n, ann->nodes_per_layer, ann->output_nodes_n, ann->learning_rate);

	cpy->memory_usage = ann->memory_usage;

	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			for (k = 0; k < ((i + 1) == (ann->hidden_layers_n + 1) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
				cpy->weights[i][j][k] = ann->weights[i][j][k];
			}
		}
	}

	return cpy;
}

void mutateANN (ANN* ann, float mutation_rate, float mutation_magnitude) {
	int i, j, k;
	int threshold = (int) ((float) RAND_MAX * mutation_rate);

	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			for (k = 0; k < ((i + 1) == (ann->hidden_layers_n + 1) ? ann->output_nodes_n : ann->nodes_per_layer); k++) {
				if (rand() < threshold) {
					if (rand() % 2) {
						ann->weights[i][j][k] += mutation_magnitude;
					} else {
						ann->weights[i][j][k] -= mutation_magnitude;
					}
				}
			}
		}
	}

	return;
}
void freeANN (ANN* ann) {
	int i, j;

	/* Weights */
	for (i = 0; i < (ann->hidden_layers_n + 1); i++) {
		for (j = 0; j < (i == 0 ? ann->input_nodes_n : ann->nodes_per_layer); j++) {
			free(ann->weights[i][j]);
		}

		free(ann->weights[i]);
	}
	free(ann->weights);

	/* Hidden layers. */
	for (i = 1; i < (ann->hidden_layers_n + 1); i++) {
		free(ann->layers[i]);
		free(ann->activation_derivatives[i]);
	}

	/* Output layer. */
	free(ann->layers[ann->hidden_layers_n + 1]);
	free(ann->activation_derivatives[ann->hidden_layers_n + 1]);

	/* Input layer */
	free(ann->layers[0]);
	free(ann->activation_derivatives[0]);

	/* Total number of layers. */
	free(ann->layers);
	free(ann->activation_derivatives);

	free(ann);

	return;
}
