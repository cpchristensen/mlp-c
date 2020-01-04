#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "MLP.h"

#define DATASET_SIZE 60000

/* Used to reverse the bits in the integer. */
int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readMNISTData (float** input, float** output, const char* filename, const char* lbl_filename, int dataset_size) {
	int i, j, magic_number, number_of_images, rows_n, cols_n, pos = 0, classes = 10;
	unsigned char byte, label;

	FILE* fp = fopen(filename, "rb");
	FILE* lbl_fp = fopen(lbl_filename, "rb");

	for (i = 0; i < (int) dataset_size; i++) {
		input[i] = malloc(28 * 28 * sizeof(float));
		output[i] = malloc(classes * sizeof(float));
		for (j = 0; j < classes; j++) {
			output[i][j] = 0.0;
		}
	}

	/* Reads header information from data. */
	fread((char*) &magic_number, sizeof(magic_number), 1, fp);
	fread((char*) &magic_number, sizeof(magic_number), 1, lbl_fp);
	magic_number = reverseInt(magic_number);
	printf("%d\n", magic_number);

	fread((char*) &number_of_images, sizeof(number_of_images), 1, fp);
	fread((char*) &number_of_images, sizeof(number_of_images), 1, lbl_fp);
	number_of_images = reverseInt(number_of_images);

	fread((char*) &rows_n, sizeof(rows_n), 1, fp);
	rows_n = reverseInt(rows_n);

	fread((char*) &cols_n, sizeof(cols_n), 1, fp);
	cols_n = reverseInt(cols_n);

	printf("%d, %d\n", rows_n, cols_n);

	while (fp) {
		fread((char*) &label, 1, 1, lbl_fp);

		output[pos][label] = 1.0;

		for (i = 0; i < (28 * 28); i++) {
			fread((char*) &byte, 1, 1, fp);
			input[pos][i] = (float) byte / 255.0;
		}

		if (++pos == (int) dataset_size) {
			break;
		}
    }

    fclose(fp);
    fclose(lbl_fp);

	return;
}

int main () {
	int i, j, k;
	int training_size = 50000, testing_size = 10000;
	int classes = 10;
	int correct, total, guess;
	float highest;
	float cost;

	ANN *ann;

    float **input = malloc(DATASET_SIZE * sizeof(float*));
    float **output = malloc(DATASET_SIZE * sizeof(float*));

	srand(time(NULL));

	printf("Reading MNIST data.\n");
	readMNISTData(input, output, "mnist-data/train-images.idx3-ubyte", "mnist-data/train-labels.idx1-ubyte", DATASET_SIZE);

	printf("Creating neural network.\n");

	ann = initANN(28 * 28, 1, 24, 10, 0.1f);
	/*ann = loadANN("mnist.ann");
	if (!ann) {
	}*/
	printf("Memory usage: %d bytes\n", ann->memory_usage);

	printf("Training neural network.\n");
	for (i = 0; i < 10; i++) {
		cost = 0.0;
		for (j = 0; j < training_size; j++) {
			cost += trainANN(ann, input[j], output[j]);
		}
		cost /= (float) training_size;

		total = correct = 0;
		for (j = training_size; j < training_size + testing_size; j++) {
			total++;

			runANN(ann, input[j]);

			highest = 0.0;
			for (k = 0; k < classes; k++) {
				if (ann->layers[ann->hidden_layers_n + 1][k] > highest) {
					highest = ann->layers[ann->hidden_layers_n + 1][k];
					guess = k;
				}
			}

			if (output[j][guess] > 0.9) {
				correct++;
			}
		}
		printf("%d: Cost: %f\tAccuracy: %.2f%%\n", i + 1, cost, ((float) correct / (float) total) * 100.0);
	}

	saveANN(ann, "mnist.ann");
	printf("Saved neural network to file.\n");

    for (i = 0; i < DATASET_SIZE; i++) {
        free(input[i]);
        free(output[i]);
    }

	free(input);
    free(output);
    freeANN(ann);

	return EXIT_SUCCESS;
}
