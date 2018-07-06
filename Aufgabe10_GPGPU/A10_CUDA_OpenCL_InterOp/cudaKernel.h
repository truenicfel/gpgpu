#ifndef CUDAKERNEL_H_
# define CUDAKERNEL_H_

// Ermittle die Größe des zu verarbeitenden Videos
// Parameter:
//    unsigned int &cols,     Zeilenbreite des Videos, Returnparameter
//    unsigned int &rows      Höhe  ""
void cudaGetOpenCVImageSize(unsigned int &cols, unsigned int &rows);

// Initialisiere die CUDA-Umgebung, übergebe dazu die von OpenGL erstellten Ressourcen
// Parameter:
//    unsigned int texId,     OpenGL-Identifier (GLuint) der von OpenGL erstellten Textur, die von CUDA mit z.B. dem Kantenbild gefüllt werden soll 
//    unsigned int vboId,     OpenGL-Identifier (GLuint) des von OpenGL erstellten Vertex Buffer Objects, in dem von CUDA 
//                            die Histogrammdaten (256 Werte, umgerechnet in Bildkoordinaten, s. main.cpp) an OpenGL übergeben werden
//    unsigned int &cols,     Zeilenbreite des Videos
//    unsigned int &rows      Höhe  ""
void cudaInit(unsigned int texId, unsigned int vboId, unsigned int cols, unsigned int rows);

// Führe einen CUDA-Berechnungsschritt aus
// Returnwert:
//    0     Es konnte ein Frame gelesen und korrekt verarbeitet werden
//   -1     Es war entweder kein Frame mehr vorhanden (Video zu Ende) oder ein Fehler ist aufgetreten
int cudaExecOneStep(unsigned char *data, size_t sz, int step, int channels, int rows, int cols);

#endif
