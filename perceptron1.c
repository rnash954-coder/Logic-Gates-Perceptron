#include <stdio.h>

int AND_inputs[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

int AND_outputs[4] = {0, 0, 0, 1};

int OR_inputs[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

int OR_outputs[4] = {0, 1, 1, 1};

int XOR_inputs[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

int XOR_outputs[4] = {0, 1, 1, 0};

int activation_function(double value)
{
    return (value >= 0.0) ? 1 : 0;
}

int predict(int x1, int x2, double w1, double w2, double bias){
    double sum;

    sum = (x1 * w1) + (x2 * w2) + bias;

    return activation_function(sum);
}

void train(int inputs[][2], int outputs[], int samples,
    double *w1, double *w2, double *bias, double learning_rate)
{
    int epoch;
    int i;
    int prediction;
    int error;

    for (epoch = 0; epoch < 100; epoch++)
    {
        int total_error = 0;

        for (i = 0; i < samples; i++)
        {
            prediction = predict(inputs[i][0], inputs[i][1],
                 *w1, *w2, *bias);

            error = outputs[i] - prediction;

            *w1 += learning_rate * error * inputs[i][0];
            *w2 += learning_rate * error * inputs[i][1];
            *bias += learning_rate * error;

            total_error += (error < 0) ? -error : error;
        }

        printf("Epoch %d, Error = %d\n",
               epoch + 1, total_error);

        if (total_error == 0)
        {
            break;
        }
    }
}

void test_gate(const char *gate_name, int inputs[][2],
     int outputs[], double w1, double w2, double bias)
{
    int i;
    int correct = 0;

    printf("\n--- %s Gate ---\n", gate_name);

    printf("Weights: w1 = %.2f, w2 = %.2f, bias = %.2f\n",
           w1, w2, bias);

    printf("\nInput  Expected  Predicted\n");

    for (i = 0; i < 4; i++)
    {
        int prediction = predict(inputs[i][0], inputs[i][1], w1, w2, bias);

        printf(" %d %d       %d          %d\n", inputs[i][0],
             inputs[i][1], outputs[i], prediction);

        if(prediction == outputs[i])
        {
            correct++;
        }
    }

    double accuracy = ((double)correct/4) * 100;

    printf("Accuracy: %.2f%%\n", accuracy);
}


int main(void)
{
    double learning_rate = 0.1;
    double AND_w1 = 0.0;
    double AND_w2 = 0.0;
    double AND_bias = 0.0;

    printf("Training AND gate\n");

    train(AND_inputs, AND_outputs, 4, &AND_w1, &AND_w2, &AND_bias, learning_rate);

    test_gate("AND", AND_inputs, AND_outputs, AND_w1, AND_w2, AND_bias);

    double OR_w1 = 0.0;
    double OR_w2 = 0.0;
    double OR_bias = 0.0;

    printf("\nTraining OR gate\n");

    train(OR_inputs, OR_outputs, 4, &OR_w1, &OR_w2, &OR_bias, learning_rate);

    test_gate("OR", OR_inputs, OR_outputs, OR_w1, OR_w2, OR_bias);

    double XOR_w1 = 0.0;
    double XOR_w2 = 0.0;
    double XOR_bias = 0.0;

    printf("\nTraining XOR gate\n");

    train(XOR_inputs, XOR_outputs, 4, &XOR_w1, &XOR_w2, &XOR_bias, learning_rate);

    test_gate("XOR", XOR_inputs, XOR_outputs, XOR_w1, XOR_w2, XOR_bias);

    return 0;
}