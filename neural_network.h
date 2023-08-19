#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

//ファイル名
#define DATASET "sin_values.csv"
#define FILENAME1 "train.csv" // トレーニング時のLossと予測値の記録用ファイル
#define FILENAME2 "test.csv" //テストデータによる検証を記録するファイル

//ファイル分割用変数
#define MAX_ROWS 100
#define MAX_LINE_LENGTH 1024
#define SPLIT_RATIO 0.8 //1.0

//ニューラルネットハイパーパラメータ(各自で設定)
#define INPUT_DATA 80 //100 //80
#define OUTPUT_DATA 80 //80
#define TEST_INPUT_DATA 20
#define TEST_OUTPUT_DATA 20

#define INPUT_NODE 1
#define INTERMEDIATE_NODE 10
#define OUTPUT_NODE 1

#define LEARNING_RATE 0.01
#define MAX_ITERATIONS 1000
#define MIN_ERROR 1e-5
#define DISPLAY_INTERVAL 10


void shuffleData(double angles[MAX_ROWS], double sin_values[MAX_ROWS], int num_rows);
//void trainAndTest(double train_angles[], double train_sin_values[], int num_train, double test_angles[], double test_sin_values[], int num_test);
//void initialize_parameters(double W1[][INTERMEDIATE_NODE], double b1[], double W2[][INTERMEDIATE_NODE], double b2[]);
#endif
