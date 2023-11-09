#include "LNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/*

*/

void InitNN(NN* n, int nb_layer, int* nb_neuron, int ActivationMode)
{
	n->NB_Layer = nb_layer;
	n->ActMode = ActivationMode;
	n->layer = (Layer*) malloc(sizeof(Layer)* nb_layer);
	for (int i = 0; i < nb_layer; i++)
	{
		n->layer[i].NB_Neuron = nb_neuron[i];
		n->layer[i].neuron = (Neuron*)malloc(sizeof(Neuron)* nb_neuron[i]);
		for (int j = 0; j < nb_neuron[i]; j++)
		{
			if(i != nb_layer - 1)
			{
				n->layer[i].neuron[j].W = malloc(sizeof(double)* nb_neuron[i + 1]);
				n->layer[i].neuron[j].dW = malloc(sizeof(double)* nb_neuron[i + 1]);
				for (int k = 0; k < nb_neuron[i + 1]; k++)
				{
					n->layer[i].neuron[j].W[k] = (double)rand() / (double)RAND_MAX;
					n->layer[i].neuron[j].dW[k] = 0.0f;
				}
			}
			else
			{
				n->layer[i].neuron[j].W = NULL;
				n->layer[i].neuron[j].dW = NULL;
			}
			n->layer[i].neuron[j].B = (double)rand() / (double)RAND_MAX;
			n->layer[i].neuron[j].Z = 0.0f;
			n->layer[i].neuron[j].A = 0.0f;
			n->layer[i].neuron[j].dA = 0.0f;
			n->layer[i].neuron[j].dB = 0.0f;
			n->layer[i].neuron[j].dZ = 0.0f;
		}
	}
	printf("%s\n", "NN Initialized ----");
}

void InitLSTM(LSTM* lstm, int nb_input, int nb_terms)
{
	lstm->nb_input = nb_input;
	lstm->nb_terms = nb_terms;
	lstm->cells = malloc(sizeof(Cell) * 4);
	lstm->LTM = calloc(nb_terms, sizeof(double));
	lstm->LTMb = calloc(nb_terms, sizeof(double));
	lstm->STM = calloc(nb_terms, sizeof(double));
	lstm->STMb = calloc(nb_terms, sizeof(double));
	for (int i = 0; i < 4; ++i)
	{
		lstm->cells[i].W = malloc(sizeof(double*) * nb_terms);
		lstm->cells[i].dW = malloc(sizeof(double*) * nb_terms);
		lstm->cells[i].B = malloc(sizeof(double) * nb_terms);
		for (int j = 0; j < nb_terms; ++j)
		{
			lstm->cells[i].W[j] = malloc(sizeof(double) * (nb_terms + nb_input));
			lstm->cells[i].dW[j] = malloc(sizeof(double) * (nb_terms + nb_input));
			for (int k = 0; k < nb_terms + nb_input; ++k)
			{
				lstm->cells[i].W[j][k] = (double)rand() / (double)RAND_MAX;
				lstm->cells[i].dW[j][k] = 0.0f;
			}
			lstm->cells[i].B[j] = (double)rand() / (double)RAND_MAX;
		}
		lstm->cells[i].Z = calloc(nb_terms, sizeof(double));
		lstm->cells[i].A = calloc(nb_terms, sizeof(double));
		lstm->cells[i].dA = calloc(nb_terms, sizeof(double));
		lstm->cells[i].dB = calloc(nb_terms, sizeof(double));
		lstm->cells[i].dZ = calloc(nb_terms, sizeof(double));
	}
}

void DestroyNN(NN* n)
{
	for (int i = 0; i < n->NB_Layer; i++)
	{
		for (int j = 0; j < n->layer[i].NB_Neuron; j++)
		{
			if(i != n->NB_Layer - 1)
			{
				free(n->layer[i].neuron[j].W);
				free(n->layer[i].neuron[j].dW);
			}
		}
		free(n->layer[i].neuron);
	}
	free(n->layer);
}

void DestroyLSTM(LSTM* lstm)
{
	free(lstm->LTM);
	free(lstm->STM);
	free(lstm->LTMb);
	free(lstm->STMb);
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < lstm->nb_terms; ++j)
		{
			free(lstm->cells[i].W[j]);
			free(lstm->cells[i].dW[j]);
		}
		free(lstm->cells[i].Z);
		free(lstm->cells[i].dZ);
		free(lstm->cells[i].A);
		free(lstm->cells[i].dA);
		free(lstm->cells[i].W);
		free(lstm->cells[i].dW);
		free(lstm->cells[i].B);
		free(lstm->cells[i].dB);
	}
	free(lstm->cells);
}


void SetNNInput(NN* n, double* input)
{
	for (int i = 0; i < n->layer[0].NB_Neuron; i++)
	{
		n->layer[0].neuron[i].A = input[i];
	}
	free(input);
}

void SetLSTMInput(LSTM* lstm, double* input)
{
	lstm->input = input;
}

void SetLSTM_LTM_STM(LSTM* lstm, double* ltm, double* stm)
{
	lstm->LTM = ltm;
	lstm->STM = stm;
}

void FeedNNForward(NN* n)
{
	for (int i = 0; i < n->NB_Layer - 1; i++)
	{
		for (int j = 0; j < n->layer[i + 1].NB_Neuron; j++)
		{
			n->layer[i + 1].neuron[j].Z = 0;
			for (int k = 0; k < n->layer[i].NB_Neuron; k++)
			{
				n->layer[i + 1].neuron[j].Z += n->layer[i].neuron[k].W[j] * n->layer[i].neuron[k].A;
			}
			n->layer[i + 1].neuron[j].Z += n->layer[i + 1].neuron[j].B;
			if(n->ActMode == FULL_SIG) n->layer[i + 1].neuron[j].A = Sigmoid(n->layer[i + 1].neuron[j].Z);
			if(n->ActMode == HIDDEN_RELU_OUT_SIG) 
			{
				if(i + 1 == n->NB_Layer - 1) n->layer[i + 1].neuron[j].A = Sigmoid(n->layer[i + 1].neuron[j].Z);
				else n->layer[i + 1].neuron[j].A = ReLu(n->layer[i + 1].neuron[j].Z);
			}
		}
	}
}

void FeedLSTMForward(LSTM* lstm)
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < lstm->nb_terms; ++j)
		{
			lstm->cells[i].Z[j] = 0.0f;
			for (int k = 0; k < lstm->nb_terms + lstm->nb_input; ++k)
			{
				if(k < lstm->nb_terms) lstm->cells[i].Z[j] += lstm->STM[k] * lstm->cells[i].W[j][k];
				else lstm->cells[i].Z[j] += lstm->input[k - lstm->nb_terms] * lstm->cells[i].W[j][k];
			}
 			lstm->cells[i].Z[j] += lstm->cells[i].B[j];
			lstm->cells[i].A[j] = i != 2 ? Sigmoid(lstm->cells[i].Z[j]) : TanH(lstm->cells[i].Z[j]);
		}
	}
	for (int i = 0; i < lstm->nb_terms; ++i)
	{
		lstm->LTMb[i] = lstm->LTM[i];
		lstm->LTM[i] = lstm->LTM[i] * lstm->cells[0].A[i] + lstm->cells[1].A[i] * lstm->cells[2].A[i];
		lstm->STMb[i] = lstm->STM[i];
		lstm->STM[i] = lstm->cells[3].A[i] * TanH(lstm->LTM[i]);
	}

}


double GetResultValue(NN n)
{
	double result = n.layer[n.NB_Layer - 1].neuron[0].A;
	for (int i = 1; i < n.layer[n.NB_Layer - 1].NB_Neuron; i++)
	{
		printf("value : %f -- index : %d\n", n.layer[n.NB_Layer - 1].neuron[i].A, i);
		if(result < n.layer[n.NB_Layer - 1].neuron[i].A)
		{
			result = n.layer[n.NB_Layer - 1].neuron[i].A;
		}
	}
	return result;
}

int GetResultIndex(NN n)
{
	int index = 0;
	double result = n.layer[n.NB_Layer - 1].neuron[0].A;
	for (int i = 1; i < n.layer[n.NB_Layer - 1].NB_Neuron; i++)
	{
		if(result < n.layer[n.NB_Layer - 1].neuron[i].A)
		{
			result = n.layer[n.NB_Layer - 1].neuron[i].A;
			index = i;
		}
	}
	return index;
}

void UpdateNNWeights(NN* n, double learning_rate)
{
	for (int i = 0; i < n->NB_Layer - 1; i++)
	{
		for (int j = 0; j < n->layer[i].NB_Neuron; j++)
		{
			for (int k = 0; k < n->layer[i + 1].NB_Neuron; k++)
			{
				n->layer[i].neuron[j].W[k] -= n->layer[i].neuron[j].dW[k] * learning_rate;
			}
		}
		
	}
}

void UpdateLSTMWeights(LSTM* lstm, double learning_rate)
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < lstm->nb_terms; ++j)
		{
			for (int k = 0; k < lstm->nb_terms + lstm->nb_input; k++)
			{
				lstm->cells[i].W[j][k] -= lstm->cells[i].dW[j][k] * learning_rate;
			}
		}
	}
}

void UpdateNNBiases(NN* n, double learning_rate)
{
	for (int i = 0; i < n->NB_Layer - 1; i++)
	{
		for (int j = 0; j < n->layer[i].NB_Neuron; j++)
		{
			n->layer[i].neuron[j].B -= n->layer[i].neuron[j].dB * learning_rate;
		}
		
	}
}

void UpdateLSTMBiases(LSTM* lstm, double learning_rate)
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < lstm->nb_terms; ++j)
		{
			lstm->cells[i].B[j] -= lstm->cells[i].dB[j] * learning_rate;
		}
		
	}
}

void TrainNN(NN* n, double* input, double* target, double learning_rate)
{
	SetNNInput(n, input);
	FeedNNForward(n);

	for (int i = n->NB_Layer - 2; i >= 0; i--)
	{
		if(i == n->NB_Layer - 2)
		{
			for (int j = 0; j < n->layer[i + 1].NB_Neuron; j++)
			{
				n->layer[i + 1].neuron[j].dZ = (n->layer[i + 1].neuron[j].A - target[j]) * n->layer[i + 1].neuron[j].A * (1 - n->layer[i + 1].neuron[j].A);
				n->layer[i + 1].neuron[j].dB = n->layer[i + 1].neuron[j].dZ;
			}
			for (int j = 0; j < n->layer[i].NB_Neuron; j++)
			{
				for (int k = 0; k < n->layer[i + 1].NB_Neuron; k++)
				{
					n->layer[i].neuron[j].dW[k] = n->layer[i + 1].neuron[k].dZ * n->layer[i].neuron[j].A;
					n->layer[i].neuron[j].dA = n->layer[i].neuron[j].W[k] * n->layer[i + 1].neuron[k].dZ;
				}
			}
		}
		else
		{
			if(n->ActMode == FULL_SIG)
			{
				for (int j = 0; j < n->layer[i + 1].NB_Neuron; j++)
				{
					n->layer[i + 1].neuron[j].dZ = n->layer[i + 2].neuron[0].dZ * n->layer[i + 1].neuron[j].W[0];
					for (int k = 1; k < n->layer[i + 2].NB_Neuron; k++)
					{
						n->layer[i + 1].neuron[j].dZ *= n->layer[i + 2].neuron[k].dZ * n->layer[i + 1].neuron[j].W[k];
					}
					n->layer[i + 1].neuron[j].dZ *= (n->layer[i + 1].neuron[j].A * (1 - n->layer[i + 1].neuron[j].A));
					n->layer[i + 1].neuron[j].dB = n->layer[i + 1].neuron[j].dZ;
				}
				for (int j = 0; j < n->layer[i].NB_Neuron; j++)
				{
					for (int k = 0; k < n->layer[i + 1].NB_Neuron; k++)
					{
						n->layer[i].neuron[j].dW[k] = n->layer[i + 1].neuron[k].dZ * n->layer[i].neuron[j].A;
					}
				}
			}
			if(n->ActMode == HIDDEN_RELU_OUT_SIG)
			{
				for (int j = 0; j < n->layer[i + 1].NB_Neuron; j++)
				{
					if(n->layer[i + 1].neuron[j].Z >= 0)
					{
						n->layer[i + 1].neuron[j].dZ = n->layer[i + 1].neuron[j].dA;
					}
					else
					{
						n->layer[i + 1].neuron[j].dZ = 0;
					}
				}
				for (int j = 0; j < n->layer[i + 1].NB_Neuron; j++)
				{
					for (int k = 0; k < n->layer[i].NB_Neuron; k++)
					{
						n->layer[i].neuron[k].dW[j] = n->layer[i + 1].neuron[j].dZ * n->layer[i].neuron[k].A;
						n->layer[i].neuron[k].dA = n->layer[i].neuron[k].W[j] * n->layer[i + 1].neuron[j].dZ;
					}
					n->layer[i + 1].neuron[j].dB = n->layer[i + 1].neuron[j].dZ;
				}
			}

		}
		
	}

	UpdateNNWeights(n, learning_rate);
	UpdateNNBiases(n, learning_rate);
}

void TrainLSTM(LSTM* lstm, double* input, double* target, double learning_rate)
{
	SetLSTMInput(lstm, input);
	FeedLSTMForward(lstm);

	for (int i = 0; i < lstm->nb_terms; ++i)
	{
		double dE = input[i] - target[i];

		lstm->cells[3].dZ[i] = dE * TanH(lstm->LTM[i]); 
		lstm->cells[2].dZ[i] = dE * lstm->cells[3].A[i] * (1 - powf(TanH(lstm->LTM[i]), 2)) * lstm->cells[1].A[i];
		lstm->cells[1].dZ[i] = dE * lstm->cells[3].A[i] * (1 - powf(TanH(lstm->LTM[i]), 2)) * lstm->cells[2].A[i];
		lstm->cells[0].dZ[i] = dE * lstm->cells[3].A[i] * (1 - powf(TanH(lstm->LTM[i]), 2)) * lstm->LTMb[i];

		for (int j = 0; j < lstm->nb_terms; ++j)
		{
			lstm->cells[3].dW[i][j] = lstm->cells[3].dZ[i] * Sigmoid(lstm->cells[3].Z[i]) * (1 - Sigmoid(lstm->cells[3].Z[i])) * lstm->STMb[j];
			lstm->cells[0].dW[i][j] = lstm->cells[0].dZ[i] * Sigmoid(lstm->cells[0].Z[i]) * (1 - Sigmoid(lstm->cells[0].Z[i])) * lstm->STMb[j];
			lstm->cells[1].dW[i][j] = lstm->cells[1].dZ[i] * Sigmoid(lstm->cells[1].Z[i]) * (1 - Sigmoid(lstm->cells[1].Z[i])) * lstm->STMb[j];
			lstm->cells[2].dW[i][j] = lstm->cells[2].dZ[i] * (1 - powf(TanH(lstm->cells[2].Z[i]), 2)) * lstm->STMb[j];
		}
		for (int j = 0; j < lstm->nb_input; ++j)
		{

			lstm->cells[3].dW[i][j + lstm->nb_terms] = lstm->cells[3].dZ[i] * Sigmoid(lstm->cells[3].Z[i]) * (1 - Sigmoid(lstm->cells[3].Z[i])) * lstm->input[j];
			lstm->cells[0].dW[i][j + lstm->nb_terms] = lstm->cells[0].dZ[i] * Sigmoid(lstm->cells[0].Z[i]) * (1 - Sigmoid(lstm->cells[0].Z[i])) * lstm->input[j];
			lstm->cells[1].dW[i][j + lstm->nb_terms] = lstm->cells[1].dZ[i] * Sigmoid(lstm->cells[1].Z[i]) * (1 - Sigmoid(lstm->cells[1].Z[i])) * lstm->input[j];
			lstm->cells[2].dW[i][j + lstm->nb_terms] = lstm->cells[2].dZ[i] * (1 - powf(TanH(lstm->cells[2].Z[i]), 2)) * lstm->input[j];
		}

		lstm->cells[3].dB[i] = lstm->cells[3].dZ[i] * Sigmoid(lstm->cells[3].Z[i]) * (1 - Sigmoid(lstm->cells[3].Z[i]));
		lstm->cells[0].dB[i] = lstm->cells[0].dZ[i] * Sigmoid(lstm->cells[0].Z[i]) * (1 - Sigmoid(lstm->cells[0].Z[i]));
		lstm->cells[1].dB[i] = lstm->cells[1].dZ[i] * Sigmoid(lstm->cells[1].Z[i]) * (1 - Sigmoid(lstm->cells[1].Z[i]));
		lstm->cells[2].dB[i] = lstm->cells[2].dZ[i] * (1 - powf(TanH(lstm->cells[2].Z[i]), 2));

	}

	
	UpdateLSTMWeights(lstm, learning_rate);
	UpdateLSTMBiases(lstm, learning_rate);
}

void DisplayNNOutput(NN n)
{
	for (int i = 0; i < n.layer[n.NB_Layer - 1].NB_Neuron; ++i)
	{
		printf("%d : %f\n", i, n.layer[n.NB_Layer - 1].neuron[i].A);
	}
}

void DisplayNNWeights(NN n)
{
	for (int i = 0; i < n.NB_Layer - 1; ++i)
	{
		for (int j = 0; j < n.layer[i].NB_Neuron; ++j)
		{
			for (int k = 0; k < n.layer[i + 1].NB_Neuron; ++k)
			{
				printf("%f ", n.layer[i].neuron[j].W[k]);
			}
			printf(",");
		}
		printf("\n");
	}
}

void DisplaySTM(LSTM l)
{
	for (int i = 0; i < l.nb_terms; ++i)
	{
		printf("Short term %d : %f\n", i, l.STM[i]);
	}
}

void SaveNNWeights(NN n, const char* filename)
{
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < n.NB_Layer - 1; ++i)
	{
		for (int j = 0; j < n.layer[i].NB_Neuron; ++j)
		{
			for (int k = 0; k < n.layer[i + 1].NB_Neuron; ++k)
			{
				fprintf(f, "%f ", n.layer[i].neuron[j].W[k]);
			}
			fprintf(f, ",");
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void LoadNNWeights(NN* n, const char* filename)
{
	FILE* f = fopen(filename, "r");
	char buffer = 0;
	char fl[20] = "\0";
	int c = 0, i = 0, j = 0, k = 0;
	while(buffer != EOF)
	{
		buffer = fgetc(f);
		if(buffer == ' ')
		{
			n->layer[i].neuron[j].W[k] = atof(fl);
			k++;
			c = 0;
		}
		if(buffer == ',')
		{
			j++;
			k = 0;
			c = 0;
		}
		if(buffer == '\n')
		{
			i++;
			j = 0;
			c = 0;
		}
		fl[c] = buffer;
		c++;
	}
	fclose(f);
}

void SaveLSTMWeights(LSTM l, const char* filename)
{
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < l.nb_terms; ++j)
		{
			for (int k = 0; k < l.nb_terms + l.nb_input; ++k)
			{
				fprintf(f, "%f ", l.cells[i].W[j][k]);
			}
			fprintf(f, "%f ", l.cells[i].B[j]);
		}
		fprintf(f, "\n");		
	}
	fclose(f);
}

void LoadLSTMWeights(LSTM* l, const char* filename)
{
	FILE* f = fopen(filename, "r");
	char buffer = 0;
	char fl[20] = "\0";
	int c = 0, i = 0, j = 0, k = 0;
	while(buffer != EOF)
	{
		buffer = fgetc(f);
		if(buffer == ' ')
		{
			if(k >= l->nb_terms + l->nb_input)
			{
				l->cells[i].B[j] = atof(fl);
				j++;
				k = 0;
			}
			else
			{
				l->cells[i].W[j][k] = atof(fl);
				k++;
			}
			c = 0;

		}
		if(buffer == '\n')
		{
			i++;
			j = 0;
			k = 0;
			c = 0;
		}
		fl[c] = buffer;
		c++;
	}
	fclose(f);
}



double Sigmoid(double value)
{
	double divv = 1.0f + expf(-value); 
	if(divv == 0.0f) return 0.0f;
	return 1.0f / divv;
}

double SigmoidDerivative(double value) 
{
    double sig = Sigmoid(value);
    return sig * (1.0f - sig);
}

double ReLu(double value)
{
	return value < 0 ? 0 : value;
}

double TanH(double value)
{
	double divv = expf(2 * value) - 1.0f;
	double divv2 = expf(2 * value) + 1.0f;
	if(divv2 == 0.0f) return 0.0f;
	return divv / divv2;
}


bitmap LoadBitmap(const char* filename, int x, int y)
{
	bitmap result;
	result.x = x;
	result.y = y;
	double** map = malloc(sizeof(double*) * x);
	FILE* fp = fopen(filename, "r");
	char buffer = 0;
	int i = 0, j = 0;
	map[i] = malloc(sizeof(double) * y);
	while(buffer != EOF)
	{
		buffer = fgetc(fp);
		if(buffer == '\n' && i < x - 1)
		{
			i++;
			map[i] = malloc(sizeof(double) * y);
			buffer = fgetc(fp);
			j = 0;
		}
		char c[2];
		c[0] = buffer;
		c[1] = '\0';
		map[i][j] = atof(c);
		j++;
	}
	fclose(fp);
	result.pixels = map;
	return result;
}

void UnloadBitmap(bitmap* bitmap)
{
	for (int i = 0; i < bitmap->x; i++)
	{
		free(bitmap->pixels[i]);
	}
	free(bitmap->pixels);
}

double* Flattening(bitmap b)
{
	int pixCount = b.x * b.y;
	double* result = malloc(sizeof(double) * pixCount);
	int c = 0;
	for (int i = 0; i < b.x; i++)
	{
		for (int j = 0; j < b.y; j++)
		{
			result[c] = b.pixels[i][j];
			c++;
		}
	}
	return result;
}
