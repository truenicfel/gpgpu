#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(float* pointerToMatrix, int matrixSize);
//void printMatrixToFile(int* pointerToMatrix, int matrixSize);

int main() {
    printf("Hello, World!\n");

    int matrixSize = 2;

    // matrix a
    // -----------------------------------------------------------------------------------------------------------------
    float* a;
    a = (float*) malloc(matrixSize * matrixSize * sizeof(float));

    if(a == NULL) {
        printf("malloc failed for matrix a.");
        exit(1);
    }

    // fill matrix a with values of counter which get incremented with every loop
    int counter = 0;
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            a[rowNumber * matrixSize + index] = counter;
            counter++;
        }
    }
    printMatrix(a,matrixSize);

    // matrix b
    // -----------------------------------------------------------------------------------------------------------------
    float* b;
    b = (float*) malloc(matrixSize * matrixSize * sizeof(float));

    if(b == NULL) {
        printf("malloc failed for matrix b.");
        exit(1);
    }
    
    // fill matrix a with values of counter which get incremented with every loop
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            b[rowNumber * matrixSize + index] = counter;
            counter--;
        }
    }
    printMatrix(b,matrixSize);

    // multiply
    // -----------------------------------------------------------------------------------------------------------------
    // result matrix c
    float* c;
    c = (float*) malloc(matrixSize * matrixSize * sizeof(float));

    if(c == NULL) {
        printf("malloc failed for matrix c.");
        exit(1);
    }

    // choose row/line
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        // choose column
        for (int columnNumber = 0; columnNumber < matrixSize; columnNumber++) {
            // result for the chose element (given by row and column)
            float result = 0;
            // iterate over both chosen row/line and column
            for (int element = 0; element < matrixSize; element++) {
                result += a[rowNumber * matrixSize + element] * b[element * matrixSize + columnNumber];
            }
            c[rowNumber * matrixSize + columnNumber] = result;
        }
    }
    printMatrix(c, matrixSize);
    return 0;
}


void printMatrix(float* pointerToMatrix, int matrixSize) {
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            printf("%f  ", pointerToMatrix[rowNumber * matrixSize + index]);
        }
        printf("\r\n");
    }
}
/*
void printMatrixToFile(int* pointerToMatrix, int matrixSize) {
    FILE* f = fopen("c.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            fprintf(f, "%d  ", pointerToMatrix[rowNumber * matrixSize + index]);
        }
        fprintf(f, "\r\n");
    }
}
*/
