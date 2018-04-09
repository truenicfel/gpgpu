#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void printMatrix(int* pointerToMatrix, int matrixSize);
void printMatrixToFile(int* pointerToMatrix, int matrixSize);

int main() {
    printf("Hello, World!\n");

    int matrixSize = 1024;

    // matrix a
    // -----------------------------------------------------------------------------------------------------------------
    int* a;
    a = (int*) malloc(matrixSize * matrixSize * sizeof(int));

    if(a == NULL) {
        printf("malloc failed for a.");
        exit(1);
    }
    int counter = 0;
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            a[rowNumber * matrixSize + index] = counter;
            counter++;
        }
    }

    // print a
    //printf("Matrix a:\r\n");
    //printMatrix(a, matrixSize);

    // matrix b
    // -----------------------------------------------------------------------------------------------------------------
    int* b;
    b = (int*) malloc(matrixSize * matrixSize * sizeof(int));

    if(b == NULL) {
        printf("malloc failed for b.");
        exit(1);
    }
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            b[rowNumber * matrixSize + index] = counter;
            counter--;
        }
    }

    // print b
    //printf("Matrix b:\r\n");
    //printMatrix(b, matrixSize);

    // multiply
    // -----------------------------------------------------------------------------------------------------------------
    // result matrix c
    int* c;
    c = (int*) malloc(matrixSize * matrixSize * sizeof(int));

    if(c == NULL) {
        printf("malloc failed for c.");
        exit(1);
    }

    // start measuring time here
    clock_t start = clock();

    // choose row/line
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        // choose column
        for (int columnNumber = 0; columnNumber < matrixSize; columnNumber++) {
            // result for the chose element (given by row and column)
            int result = 0;
            // iterate over both chosen row/line and column
            for (int element = 0; element < matrixSize; element++) {
                result += a[rowNumber * matrixSize + element] * b[element * matrixSize + columnNumber];
            }
            c[rowNumber * matrixSize + columnNumber] = result;
        }
    }

    // end time measurement here
    clock_t stop = clock();

    double elapsed = (double)(stop - start)/ CLOCKS_PER_SEC;

    printf("Time: %f", elapsed);

    // print c
    //printf("Matrix c:\r\n");
    printMatrixToFile(c, matrixSize);

    return 0;
}

void printMatrix(int* pointerToMatrix, int matrixSize) {
    for (int rowNumber = 0; rowNumber < matrixSize; rowNumber++) {
        for (int index = 0; index < matrixSize; index++) {
            printf("%d  ", pointerToMatrix[rowNumber * matrixSize + index]);
        }
        printf("\r\n");
    }
}

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
