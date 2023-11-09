#pragma once
#ifndef __LNN__
#define __LNN__

#define FULL_SIG 0
#define HIDDEN_RELU_OUT_SIG 1

typedef struct NEURON
{
	double A;
	double Z;
	double* W;
	double B;

	double dA;
	double dZ;
	double* dW;
	double dB;
} Neuron;

typedef struct LSTMCELL
{
	double* A;
	double* Z;
	double** W;
	double* B;

	double* dA;
	double* dZ;
	double** dW;
	double* dB;
} Cell;

typedef struct LAYER
{
	int NB_Neuron; 
	Neuron* neuron;
} Layer;

typedef struct NeuralNetwork
{
	int NB_Layer;
	Layer* layer;
	int ActMode;
} NN;

typedef struct LSTM
{
	Cell* cells;
	double* LTM;
	double* LTMb;
	double* STM;
	double* STMb;
	double* input;
	int nb_input;
	int nb_terms;

} LSTM;

typedef struct BITMAP
{
	double** pixels;
	int x;
	int y;
}bitmap;



double Sigmoid(double value);
double SigmoidDerivative(double value);
double ReLu(double value);
double TanH(double value);


void InitNN(NN* n, int nb_layer, int* nb_neuron, int ActMode);
void DestroyNN(NN* n);
void SetNNInput(NN* n, double* input);
void FeedNNForward(NN* n);
double GetResultValue(NN n);
int GetResultIndex(NN n);
void TrainNN(NN* n, double* input, double* target, double learning_rate);
void UpdateNNWeights(NN* n, double learning_rate);
void UpdateNNBiases(NN* n, double learning_rate);
void SaveNNWeights(NN n, const char* filename);
void LoadNNWeights(NN* n, const char* filename);
void DisplayNNWeights(NN n);
void DisplayNNOutput(NN n);


void InitLSTM(LSTM* lstm, int nb_input, int nb_terms);
void DestroyLSTM(LSTM* lstm);
void SetLSTMInput(LSTM* lstm, double* input);
void FeedLSTMForward(LSTM* lstm);
void UpdateLSTMWeights(LSTM* lstm, double learning_rate);
void UpdateLSTMBiases(LSTM* lstm, double learning_rate);
void TrainLSTM(LSTM* lstm, double* input, double* target, double learning_rate);
void SaveLSTMWeights(LSTM l, const char* filename);
void LoadLSTMWeights(LSTM* l, const char* filename);
void DisplaySTM(LSTM l);



bitmap LoadBitmap(const char* filename, int x, int y);
void UnloadBitmap(bitmap* bitmap);
double* Flattening(bitmap b);

#endif