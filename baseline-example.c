#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "genann/genann.h"

#define DATASET_SIZE 40000

/* Used to reverse the bits in the integer. */
int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void readMNISTData (double** input, double** output, const char* filename, const char* lbl_filename, int dataset_size) {
	int i, j, magic_number, number_of_images, rows_n, cols_n, pos = 0, classes = 10;
	unsigned char byte, label;

	FILE* fp = fopen(filename, "rb");
	FILE* lbl_fp = fopen(lbl_filename, "rb");

	for (i = 0; i < (int) dataset_size; i++) {
		input[i] = malloc(28 * 28 * sizeof(double));
		output[i] = malloc(classes * sizeof(double));
		for (j = 0; j < classes; j++) {
			output[i][j] = 0.0;
		}
	}

	/* Reads header information from data. */
	fread((char*) &magic_number, sizeof(magic_number), 1, fp);
	fread((char*) &magic_number, sizeof(magic_number), 1, lbl_fp);
	magic_number = reverseInt(magic_number);

	fread((char*) &number_of_images, sizeof(number_of_images), 1, fp);
	fread((char*) &number_of_images, sizeof(number_of_images), 1, lbl_fp);
	number_of_images = reverseInt(number_of_images);

	fread((char*) &rows_n, sizeof(rows_n), 1, fp);
	rows_n = reverseInt(rows_n);

	fread((char*) &cols_n, sizeof(cols_n), 1, fp);
	cols_n = reverseInt(cols_n);

	printf("%d, %d\n", rows_n, cols_n);

	while (fp) {
		fread(&label, 1, 1, lbl_fp);

		output[pos][label] = 1.0;

		for (i = 0; i < (28 * 28); i++) {
			fread((char*) &byte, 1, 1, fp);
			input[pos][i] = (double) byte / 255.0;
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
	int training_size = 30000, testing_size = 5000;
	int classes = 10;
	int correct, total, guess;
	double highest;

	genann* ann;

    double **input = malloc(DATASET_SIZE * sizeof(double*));
    double **output = malloc(DATASET_SIZE * sizeof(double*));

	srand(time(NULL));

	printf("Reading MNIST data.\n");
	readMNISTData(input, output, "mnist-data/train-images.idx3-ubyte", "mnist-data/train-labels.idx1-ubyte", DATASET_SIZE);

	printf("Creating neural network.\n");

	ann = genann_init(28 * 28, 2, 16, 10);

	printf("Training neural network.\n");
	for (i = 0; i < 10; i++) {
		for (j = 0; j < training_size; j++) {
			genann_train(ann, input[j], output[j], 1.0);
		}

		total = correct = 0;
		for (j = training_size; j < training_size + testing_size; j++) {
			total++;

			const double* ret = genann_run(ann, input[j]);

			highest = 0.0;
			for (k = 0; k < classes; k++) {
				if (ret[k] > highest) {
					highest = ret[k];
					guess = k;
				}
			}

			if (output[j][guess] > 0.99) {
				correct++;
			}

		}
		printf("%d: Accuracy: %.2f%%\n", i + 1, ((double) correct / (double) total) * 100.0);
	}

    for (i = 0; i < DATASET_SIZE; i++) {
        free(input[i]);
        free(output[i]);
    }

	free(input);
    free(output);
    genann_free(ann);

	return EXIT_SUCCESS;
}
