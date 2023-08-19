#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neural_network.h"

// シグモイド関数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Relu関数
double ReLU(double x) {
    if (x < 0) {
        return 0;
    } else {
        return x;
    }
}

void shuffleData(double angles[MAX_ROWS], double sin_values[MAX_ROWS], int num_rows) {
    srand(time(NULL));

    for (int i = num_rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        double temp_angle = angles[i];
        double temp_sin = sin_values[i];

        angles[i] = angles[j];
        sin_values[i] = sin_values[j];

        angles[j] = temp_angle;
        sin_values[j] = temp_sin;
    }
}

/*-----------------------------重み・バイアス初期値生成------------------------------------------*/

// 重みとバイアスの初期化関数
void initialize_parameters(double W1[][INTERMEDIATE_NODE], double b1[], double W2[][INTERMEDIATE_NODE], double b2[]) {
    srand(time(NULL));

    // W1の初期化
    for (int i = 0; i < INPUT_DATA; ++i) {
        for (int j = 0; j < INTERMEDIATE_NODE; ++j) {
            W1[i][j] = (double)rand() / RAND_MAX - 0.5; // -0.5から0.5の範囲の乱数
        }
    }

    // b1の初期化
    for (int i = 0; i < INPUT_DATA; ++i) {
        b1[i] = (double)rand() / RAND_MAX - 0.5; // -0.5から0.5の範囲の乱数
    }

    // W2の初期化
    for (int i = 0; i < OUTPUT_DATA; ++i) {
        for (int j = 0; j < INTERMEDIATE_NODE; ++j) {
            W2[i][j] = (double)rand() / RAND_MAX - 0.5; // -0.5から0.5の範囲の乱数
        }
    }

    // b2の初期化
    for (int i = 0; i < OUTPUT_DATA; ++i) {
        b2[i] = (double)rand() / RAND_MAX - 0.5; // -0.5から0.5の範囲の乱数
    }
}

/*-----------------------------重み・バイアス初期値の表示------------------------------------------*/
/* 重みとバイアスの初期値を表示する関数
void displayInitialParameters(double W1[][INTERMEDIATE_NODE], double b1[], double W2[][INTERMEDIATE_NODE], double b2[]) {
    printf("\nInitial Weights and Biases:\n");

    printf("W1:\n");
    for (int i = 0; i < INPUT_DATA; i++) {
        printf("サンプル%dの重み1\n",i+1);
        for (int j = 0; j < INTERMEDIATE_NODE; j++) {
            printf("%.2lf ",W1[i][j]);
        }
        printf("\n");
    }

    printf("\nb1:\n");
    for (int i = 0; i < INPUT_DATA; i++) {
        printf("%.2lf ", b1[i]);
    }
    printf("\n");

    printf("\nW2:\n");
    for (int i = 0; i < INTERMEDIATE_NODE; i++) {
        printf("サンプル%dの重み2\n",i+1);
        for (int j = 0; j < OUTPUT_DATA; j++) {
            printf("%.2lf ", W2[i][j]);
        }
        printf("\n");
    }

    printf("\nb2:\n");
    for (int i = 0; i < OUTPUT_DATA; i++) {
        printf("%.2lf ", b2[i]);
    }
    printf("\n");
}*/


int main() {
    // ファイルを開く
    FILE *file = fopen( DATASET , "r");
    if (file == NULL) {
        perror("ファイルを開けません");
        return 1;
    }

// Lossと予測値の記録用ファイル
FILE *csv_file = fopen(FILENAME1, "w");
if (csv_file == NULL) {
    perror("Failed to create CSV file");
    return 1;
}

fprintf(csv_file, " epoch, loss[i], predict, sin\n");


    // データを格納する配列
    double angles[MAX_ROWS];
    double sin_values[MAX_ROWS];

    char line[MAX_LINE_LENGTH];
    int row = 0;

    // ヘッダ行を読み飛ばす
    fgets(line, sizeof(line), file);

    // データの読み込みと格納
    while (row < MAX_ROWS && fgets(line, sizeof(line), file) != NULL) {
        char *token = strtok(line, ",");
        int col = 0;

        while (token != NULL) {
            if (col == 0) {
                angles[row] = atof(token);
            } else if (col == 2) {
                sin_values[row] = atof(token);
            }

            token = strtok(NULL, ",");
            col++;
        }

        row++;
    }

    // ファイルを閉じる
    fclose(file);

// シャッフルデータ
    //shuffleData(angles, sin_values, row);

    // トレーニングデータとテストデータの配列を作成
    double train_angles[INPUT_DATA];//入力データ格納用
    double train_sin_values[OUTPUT_DATA];//教師データ
    double test_angles[MAX_ROWS];//テスト用入力値
    double test_sin_values[MAX_ROWS];//正解値


// トレーニングデータとテストデータの割合
    double train_ratio = SPLIT_RATIO; // トレーニングデータの割合
    int num_train = (int)(row * train_ratio);
    int num_test = row - num_train;

    // トレーニングデータを分割
    for (int i = 0; i < num_train; i++) {
        train_angles[i] = angles[i];
        train_sin_values[i] = sin_values[i];
    }

    // テストデータを分割
    for (int i = num_train; i < row; i++) {
        test_angles[i - num_train] = angles[i];
        test_sin_values[i - num_train] = sin_values[i];
    }

    /* トレーニングデータを表示
    printf("Training Data:\n");
    for (int i = 0; i < num_train; i++) {
        printf("Angle: %.2lf, Sin: %.2lf\n", train_angles[i], train_sin_values[i]);
    }

    // テストデータを表示
    printf("\nTest Data:\n");
    for (int i = 0; i < num_test; i++) {
        printf("Angle: %.2lf, Sin: %.2lf\n", test_angles[i], test_sin_values[i]);
    }*/

    // 重みとバイアスの初期化
    double W1[INPUT_DATA][INTERMEDIATE_NODE]; //入力層→中間層の重み
    double b1[INPUT_DATA]; //入力層→中間層のバイアス
    double W2[OUTPUT_DATA][INTERMEDIATE_NODE]; //中間層→出力層の重み
    double b2[OUTPUT_DATA]; //中間層→出力層のバイアス

    initialize_parameters(W1, b1, W2, b2);
    // 初期化された重みとバイアスの表示
    //displayInitialParameters(W1, b1, W2, b2);

    double a1[INPUT_DATA][INTERMEDIATE_NODE];
    double z2[OUTPUT_DATA][INTERMEDIATE_NODE];
    double a2[INPUT_DATA];
    double output[OUTPUT_DATA];

    double delta2[OUTPUT_DATA];
    double dW2[OUTPUT_DATA][INTERMEDIATE_NODE];
    double db2[INPUT_DATA];
    double delta1[INPUT_DATA][INTERMEDIATE_NODE];
    double dW1[INPUT_NODE][INTERMEDIATE_NODE];
    double db1[INPUT_DATA];

/*---------------------------------トレーニングフェーズ--------------------------------------*/


// 学習ループ
    for (int epoch = 0; epoch < 1000; ++epoch) {

// 順伝播
    //入力層〜中間層
    //printf("\n");
    //printf("中間層結果\n");

    for (int i = 0; i < INPUT_DATA; ++i) {
        //z1 = 0;
        //printf("\n");
        //printf("サンプル%d\n",i+1);
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        /*z1 = train_angles[i] * W1[i][j] + b1[i];
        a1[i][j] = sigmoid(z1);*/
        a1[i][j] = sigmoid(train_angles[i] * W1[i][j] + b1[i]); //sigmoid ReLU
        //printf("ノード%dの出力値 : %f\n",j+1,a1[i][j]);
        }
    }

    //printf("\n");
    //printf("出力層結果（出力値）\n");

    //中間層〜出力層
    for (int i = 0; i < INPUT_DATA; ++i) {
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        a2[i] += a1[i][j] * W2[i][j] + b2[i];
        }
    }

    for(int i = 0; i < INPUT_DATA; ++i){
        output[i] = a2[i];
        //printf("サンプル%dの予測値 : %f\n",i+1,output[i]);
    }

// 損失の計算
    double loss[INPUT_DATA];
    for (int i = 0; i < INPUT_DATA; ++i) {
        loss[i] = pow(output[i] - train_sin_values[i], 2);
        //printf("サンプル%dのLoss: %f\n",i+1,loss[i]);
    }
    //loss[] /= INPUT_DATA;
    //printf("\n");


// 逆伝播

    for (int i = 0; i < INPUT_DATA; ++i) {
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        delta2[i] = output[i] - train_sin_values[i];
        dW2[i][j] += a1[i][j] * delta2[i];
        delta1[i][j] = delta2[i] * W2[i][j] * a1[i][j] * (1.0 - a1[i][j]);
        dW1[i][j] += train_angles[i] * delta1[i][j];

        // パラメータの更新
        W1[i][j] -= LEARNING_RATE * dW1[i][j] / INPUT_DATA;
        b1[i] -= LEARNING_RATE * db1[i] / INPUT_DATA;
        W2[i][j] -= LEARNING_RATE * dW2[i][j] / INPUT_DATA;
        b2[i] -= LEARNING_RATE * db2[i] / INPUT_DATA;
        }
    }

        if (epoch % 100 == 0) {
            //printf("\n");
            //printf("Epoch %d\n", epoch);
            for (int i = 0; i < INPUT_DATA; ++i) {
            //printf("サンプル%dのLoss: %f\n", i+1,loss[i]);
            //printf("サンプル%dの予測値: %f\n", i+1,output[i]);

            fprintf(csv_file, "%d,%.6lf,%.6lf,%.6lf\n", epoch, loss[i], output[i], train_sin_values[i]);

            //fprintf(updates_file,"%d, %f\n", epoch, loss);
            // パラメータの更新の値をファイルに記録
            //fprintf(updates_file, "%d,%f,%f,%f,%f\n", epoch, dW1[0][0], db1[0], dW2[0][0], db2[0]);
        }
        }

//a1・a2の初期化
    for(int i = 0; i < INPUT_DATA; ++i){
        a2[i] = 0;
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        a1[i][j] = 0;
    }
    }
}
fclose(csv_file);
/*-----------------------------テストフェーズ-----------------------*/
//学習済みモデルで予測
    double predicted_output[TEST_OUTPUT_DATA];
    // テスト用 順伝播

    for (int i = 0; i < TEST_INPUT_DATA; ++i) {
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        a1[i][j] = sigmoid(test_angles[i] * W1[i][j] + b1[i]); //sigmoid ReLU
        }
    }

    //中間層〜出力層
    for (int i = 0; i < TEST_OUTPUT_DATA; ++i) {
        for(int j = 0; j < INTERMEDIATE_NODE; ++j){
        a2[i] += a1[i][j] * W2[i][j] + b2[i];
        }
    }

    for(int i = 0; i < TEST_OUTPUT_DATA; ++i){
        predicted_output[i] = a2[i];
        //printf("サンプル%dの予測値 : %f\n",i+1,output[i]);
    }

    // テスト用 損失の計算
    double error[TEST_OUTPUT_DATA];
    for (int i = 0; i < TEST_OUTPUT_DATA; ++i) {
        error[i] = pow(predicted_output[i] - test_sin_values[i], 2);
        //printf("サンプル%dのLoss: %f\n",i+1,error[i]);
    }

    //正解値と出力値の誤差の表示
    //CSVファイルに記録
    FILE *csv_file2 = fopen( FILENAME2 , "w");
    if (csv_file == NULL) {
        perror("ファイルを開けませんでした");
        return 1;
    }

    printf("入力値\t\t正解値\t\t出力値\t\t誤差\n");
    fprintf(csv_file2, "Test_input,Test_data,predicted,ERROR\n");
    for (int i = 0; i < TEST_OUTPUT_DATA; ++i) {
        printf("%f\t%f\t%f\t%f\n", test_angles[i], test_sin_values[i], predicted_output[i], error[i]);
        fprintf(csv_file2, "%f,%f,%f,%f\n",  test_angles[i], test_sin_values[i], predicted_output[i], error[i]);
    }

    fclose(csv_file2);


// After the training loop


    return 0;
}

