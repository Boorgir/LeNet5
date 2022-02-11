#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include "weights_n_bias.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define WIDTH 28
#define HEIGHT 28


void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

void ReadImageFromFile(int ***img, FILE* fptr){
    int color[3]={255,0,0};
    for(int i=0; i<HEIGHT; i++){
        for(int j=0; j<WIDTH; j++){ 
            unsigned char val;
            fread(&val, sizeof(unsigned char), 1, fptr);  
            img[i][j][0]=(int)val*color[0]/255;
            img[i][j][1]=(int)val*color[1]/255;
            img[i][j][2]=(int)val*color[2]/255;
        }
    }
}





__device__ float activation_tanh(float M){
    //fonction d'activation tanh qui applique à un float la fonction tanh
    return tanhf(M);
}


__host__ void activation_softmax(float *M_in, float *M_out, int n){
    //M_in est la matrice des éléments à qui on va appliquer softmax
    //M_out est la matrice dans laquelle on va stocker les éléments de sortie
    //n est la longueur de la matrice
    //la fonction d'activation softmax correspond à l'exponentielle du coefficient sur la somme des exponentielles de tous les coefficients
    
    float tot = 0;

    for(int k=0;k<n;k++){
        tot = tot + expf(M_in[k]);
    }

    for(int k=0;k<n;k++){
        M_out[k] = expf(M_in[k])/tot;
    }

}



__host__ void MatrixInit2(float *M, int n, int p, int mini, int maxi){
    //on initialise la matrice M de taille nxp avec des nombres flotants aléatoires entre mini et maxi
    for(int k=0;k<n*p;k++){
            M[k]=((float)rand())/RAND_MAX*(maxi-mini)+mini;
    }
}

__host__ void MatrixInit(float *M, int n, int p){
    //on initialise la matrice M de taille nxp avec des nombres flotants aléatoires entre 0 et 1    
    for(int k=0;k<n*p;k++){
            M[k]=((float)rand())/RAND_MAX*2-1;
    }
}

__host__ void InitKernel(float *M, int n, int p){
    //M est la matrice qu'on initialise, elle est de taille nxp
    //on initialise le kernel pour pour faire des tests
    //un 1 au centre et des 0 autour : c'est la matrice identité pour la convolution
    for(int k=0;k<n*p;k++){
            M[k]=0;
    }
    for(int a=0;a<6;a++){
        M[12 + 25*a]=1;
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, float *biais){
    //on mutipli les matrices M1 et M2 et on stock le résultat dans la matrice Mout
    //n est la dimension en commun entre les deux matrices
    // un biais est ajouté à chaque neuronne de sortie
    //on active avec tanh à la fin
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    

    float a = 0;
    for(int k=0;k<n;k++){
        a = a + M1[i*n+k]*M2[n*k+j];
    }
    
    Mout[i*n + j] = activation_tanh(a + biais[j]);
}

__global__ void cudaMatrixMult2(float *M1, float *M2, float *Mout, int n, float *biais){
    //on mutipli les matrices M1 et M2 et on stock le résultat dans la matrice Mout
    //n est la dimension en commun entre les deux matrices
    // un biais est ajouté à chaque neuronne de sortie
    // il n'y a pas d'activation à la fin
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    

    float a = 0;
    for(int k=0;k<n;k++){
        a = a + M1[i*n+k]*M2[n*k+j];
    }
    
    Mout[i*n + j] = a + biais[j];
}



__host__ void MatrixPrint2(float *M, int n, int p, int m){
    //permet d'afficher des matrices à 3 dimensions
    
    for(int k=0;k<m;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<p;j++){
                if(M[i*p + j]>=0) printf(" ");
                printf("%1.1f ",M[i*p + j + k*n*p]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

__host__ void MatrixPrint(float *M, int n, int p){
    //permet d'afficher les matrices à 2 dimensions
    
    for(int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            if(M[i*p + j]>0) printf(" ");
            printf("%1.1f ",M[i*p + j]);
        }
        printf("\n");
    }
    printf("\n");
}




__global__ void cudaConvol(float *raw_data, float *C1_kernel, float *C1_data, int nk, int n, float *biais){
    // la convolution c'est le premier on y touche pas, le second on fait une symétrie centrale de la matrice (retournée horrizontalement puis verticalement) , on multiplie terme à terme et on somme le tout
    //nk taille d'un élément du kernel ici 5
    //est la taille d'un matrice, ici 28
    //n est 28
    //un biais est ajouté à chaque neuronne
    int ligne = threadIdx.x; // 0 -> 27
    int colonne = threadIdx.y; // 0 -> 27
    int kernel = blockIdx.x; // 0 -> 5 
    
    float a = 0;
    for(int q=0;q<nk;q++){
        for(int k=0;k<nk;k++){
            a += raw_data[ (ligne+q)*(n+nk-1) + k+colonne ] * C1_kernel[kernel*nk*nk + nk*(nk-1-q) + (nk-1-k)];
        }
    }
   
    C1_data[kernel*n*n + ligne*n + colonne] = activation_tanh(a + biais[kernel]);
}

__global__ void cudaMaxPool(int n, int nf, int nfiltre, float *MIn, float *MOut){
//n est la taille de la matrice initiale, c'est à dire 28
//nfiltre est la taille de la fenêtre à appliquer
    //nf taille de la matrice finale
    int ligne = threadIdx.x; // se balade entre 1 et 14
    int colonne = threadIdx.y; // se balade entre 1 et 14
    int kernel = blockIdx.x; //0->6
    
    
    
    float a = MIn[kernel*n*n + ligne*n*2 + colonne*2];

    if(a<MIn[kernel*n*n + ligne*n*2 + colonne*2+1]){
        a = MIn[kernel*n*n + ligne*n*2 + colonne*2+1];
    }
    if(a<MIn[kernel*n*n + ligne*n*2+n + colonne*2+1]){
        a=MIn[kernel*n*n + ligne*n*2+n + colonne*2+1];
    }
    if(a<MIn[kernel*n*n + ligne*n*2+n + colonne*2]){
        a=MIn[kernel*n*n + ligne*n*2+n + colonne*2];
    }
    

    MOut[kernel*nf*nf + ligne*nf + colonne] = a;
    
}



__global__ void cudaAveragePooling2x2(int n, int nf, float *MIn, float *MOut){
    //n = Size of initial matrix (28)
    //nf = Size of resulting matrix (14)
    int ligne = threadIdx.x; // 0 -> 13
    int colonne = threadIdx.y; // 0 -> 13
    int kernel = blockIdx.x; //0->5
    
    float a = 0;
    
    a = 0.25*(MIn[kernel*n*n + ligne*2*n + colonne*2] + MIn[kernel*n*n + ligne*2*n + colonne*2+1] + MIn[kernel*n*n + (ligne*2+1)*n + colonne*2] + MIn[kernel*n*n + (ligne*2+1)*n + colonne*2+1]);


    MOut[kernel*nf*nf + ligne*nf + colonne] = activation_tanh(a);
    
}

__global__ void cudaConvol2(float *raw_data, float *C1_kernel, float *C1_data, int nk, int n, float *biais){
    //cette convolution modifiée prend un kernel 16x6x5x5
    // la convolution c'est le premier on y touche pas, le second on fait une symétrie centrale de la matrice (retournée horrizontalement puis verticalement) , on multiplie terme à terme et on somme le tout
    //nk taille d'un élément du kernel ici 5
    //est la taille d'un matrice, ici 28
    //n est 28
    //un biais est ajouté à chaque neuronne
    int ligne = threadIdx.x; // 0 -> 9
    int colonne = threadIdx.y; // 0 -> 9
    int kernel = blockIdx.x; // 0 -> 15 

    
    float a = 0;
    for(int nb_k=0; nb_k<6; nb_k++){ //on fait la somme sur 6 kernel
        for(int q=0;q<nk;q++){
            for(int k=0;k<nk;k++){
                a += raw_data[ (ligne+q)*(n+nk-1) + k+colonne ] * C1_kernel[kernel*6*nk*nk + nb_k*nk*nk + nk*(nk-1-q) + (nk-1-k)]  + biais[kernel*6 + nb_k];
            }
        }
    }

    C1_data[kernel*n*n + ligne*n + colonne] = activation_tanh(a);
}

void init(float *M1, float *d_M1, int size_h, int size_w, int size_l, int min, int max, int kernel){
    //cette fonction ne marche pas du tout, les malloc ne traverse pas les fonctions
    // Allocate memory
    M1 = (float*)malloc(sizeof(float) * size_h*size_w*size_l);
    
    if(kernel==0){
        MatrixInit2(M1, size_h*size_w, size_l, min, max);
    }
    else{
        InitKernel(M1, size_h*size_w, size_l);
    }
    
    
    cudaMalloc((void**)&d_M1, sizeof(float)*size_h*size_w*size_l);
    cudaMemcpy(d_M1, M1, sizeof(float) * size_h*size_w*size_l, cudaMemcpyHostToDevice);
}



int main(int argc, char *argv[]){
    srand(time(NULL));
    
    
  int i, j;
  int ***img;
  unsigned int magic, nbImg, nbRows, nbCols;
  FILE *fptr;

  // Malloc image
  img = (int ***)malloc(HEIGHT*sizeof(int **));
  for(i=0; i<HEIGHT; i++){
    img[i]= (int **)malloc(WIDTH*sizeof(int *));
    for(j=0; j<WIDTH; j++){
      img[i][j] = (int *)malloc(sizeof(int)*3);
    }
  }

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);
  
    
  //on prend une image au hasard dans le dataset et on l'affiche
  int a=(int)(rand()/RAND_MAX*10 +1);
  printf("%i \n",a);
  for(int k=0;k<((int)rand());k++){
      ReadImageFromFile(img,fptr);
  }
  
  imgColorPrint(HEIGHT, WIDTH, img);
    
    
    
   
    
    
    
    
    
    
    
 
    //Création des matrices
    
    float *raw_data; //taille 32x32 entre 0 et 1
    float *C1_data; //6X28X28 à 0
    float *S1_data; //6x14x14 à 0
    float *C1_kernel; //6x5x5 initialisé entre 0 et 1
    float *C3_kernel;//16x6x5x5
    float *S3_data; //10x10x16
    float *S4_data; //5x5x16
    float *S5_data; //1x120
    float *D1_dense; //400x120
    float *S6_data; //1x84
    float *D2_dense; //120x84
    float *S7_data; //1x10
    float *D3_dense; //84x10
    float *S8_data; //1x10
    
    float *biais1; //6
    float *biais2; //16
    float *biais3; //120
    float *biais4; //84
    float *biais5; //10
    
    
    
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel, *d_C3_kernel, *d_S3_data, *d_S4_data, *d_S5_data, *d_D1_dense, *d_S6_data, *d_D2_dense, *d_S7_data, *d_D3_dense, *d_S8_data;
    
    float *d_biais1, *d_biais2, *d_biais3, *d_biais4, *d_biais5;
    
    
    //initialisation des matrices
    
    //init(raw_data, d_raw_data, 32, 32, 1, 0, 1, 0);
    raw_data = (float*)malloc(sizeof(float) * 32*32*1);
    
    MatrixInit2(raw_data, 32*32, 1, 0, 0); //initialise à 0 poure faire du 0 padding
    
    for(int i=0; i<HEIGHT; i++){
        for(int j=0; j<WIDTH; j++){ 
            raw_data[(i+2)*32+(j+2)] = ((float)img[i][j][0])/255;
        }
    }
    
    //MatrixPrint2(raw_data,32,32,1);

    
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32*32*1);
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32*32*1, cudaMemcpyHostToDevice);
    
    //init(C1_data, d_C1_data, 28, 28, 6, 0, 0, 0);
    C1_data = (float*)malloc(sizeof(float) * 28*28*6);
    MatrixInit2(C1_data, 28*28, 6, 0, 1);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28*28*6);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28*28*6, cudaMemcpyHostToDevice);    

    
    //init(S1_data, d_S1_data, 14, 14, 6, 0, 0, 0);
    S1_data = (float*)malloc(sizeof(float) * 14*14*6);
    MatrixInit2(S1_data, 14*14, 6, 0, 0);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14*14*6);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14*14*6, cudaMemcpyHostToDevice);    
    
    //init(C1_kernel, d_C1_kernel, 5, 5, 6, 0, 0, 1);
    C1_kernel = (float*)malloc(sizeof(float) * 5*5*6);
    C1_kernel = Conv1_weights;
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5*5*6);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5*5*6, cudaMemcpyHostToDevice);    
    
    //biais de la première convolution
    biais1 = (float*)malloc(sizeof(float) * 6*1);
    biais1 = Conv1_bias;
    cudaMalloc((void**)&d_biais1, sizeof(float) * 6*1);
    cudaMemcpy(d_biais1, biais1, sizeof(float) * 6*1, cudaMemcpyHostToDevice);
    
    
    //init(C3_kernel, d_C3_kernel, 5, 5, 16*6, 0, 0, 1);
    C3_kernel = (float*)malloc(sizeof(float) * 5*5*16*6);
    C3_kernel = Conv2_weights;
    cudaMalloc((void**)&d_C3_kernel, sizeof(float) * 5*5*16*6);
    cudaMemcpy(d_C3_kernel, C3_kernel, sizeof(float) * 5*5*16*6, cudaMemcpyHostToDevice);    
    
    
    //biais de la deuxième convolution
    biais2 = (float*)malloc(sizeof(float) * 16*1);
    biais2 = Conv2_bias;
    cudaMalloc((void**)&d_biais2, sizeof(float) * 16*1);
    cudaMemcpy(d_biais2, biais2, sizeof(float) * 16*1, cudaMemcpyHostToDevice);
    
    
    
    //init(S3_data, d_S3_data, 10, 10, 16, 0, 0, 0);
    S3_data = (float*)malloc(sizeof(float) * 10*10*16);
    MatrixInit2(S3_data, 10*10, 16, 0, 0);
    cudaMalloc((void**)&d_S3_data, sizeof(float) * 10*10*16);
    cudaMemcpy(d_S3_data, S3_data, sizeof(float) * 10*10*16, cudaMemcpyHostToDevice);     
    
    //init(S4_data, d_S4_data, 5, 5, 16, 0, 0, 0);
    S4_data = (float*)malloc(sizeof(float) * 5*5*16);
    MatrixInit2(S4_data, 5*5, 16, 0, 0);
    cudaMalloc((void**)&d_S4_data, sizeof(float) * 5*5*16);
    cudaMemcpy(d_S4_data, S4_data, sizeof(float) * 5*5*16, cudaMemcpyHostToDevice);     
    
    //init(S5_data, d_S5_data, 1, 120, 1, 0, 0, 0);
    S5_data = (float*)malloc(sizeof(float) * 1*120);
    MatrixInit2(S5_data, 1, 120, 0, 0);
    cudaMalloc((void**)&d_S5_data, sizeof(float) * 1*120);
    cudaMemcpy(d_S5_data, S5_data, sizeof(float) * 1*120, cudaMemcpyHostToDevice);         
    
    //init(D1_dense, d_D1_dense, 400, 120, 1, 0, 0, 0);
    D1_dense = (float*)malloc(sizeof(float) * 400*120);
    D1_dense = Dense1_weights;
    cudaMalloc((void**)&d_D1_dense, sizeof(float) * 400*120);
    cudaMemcpy(d_D1_dense, D1_dense, sizeof(float) * 400*120, cudaMemcpyHostToDevice);    
    
    
    //biais du premier dense
    biais3 = (float*)malloc(sizeof(float) * 120*1);
    biais3 = Dense1_bias;
    cudaMalloc((void**)&d_biais3, sizeof(float) * 120*1);
    cudaMemcpy(d_biais3, biais3, sizeof(float) * 120*1, cudaMemcpyHostToDevice);
    
    //init(S6_data, d_S6_data, 1, 84, 1, 0, 0, 0);
    S6_data = (float*)malloc(sizeof(float) * 1*120);
    MatrixInit2(S6_data, 1, 84, 0, 0);
    cudaMalloc((void**)&d_S6_data, sizeof(float) * 1*84);
    cudaMemcpy(d_S6_data, S6_data, sizeof(float) * 1*84, cudaMemcpyHostToDevice);       
    
    //init(D2_dense, d_D2_dense, 120, 84, 1, 0, 0, 0);
    D2_dense = (float*)malloc(sizeof(float) * 84*120);
    D2_dense = Dense2_weights;
    cudaMalloc((void**)&d_D2_dense, sizeof(float) * 84*120);
    cudaMemcpy(d_D2_dense, D2_dense, sizeof(float) * 84*120, cudaMemcpyHostToDevice); 
    
    //biais du deuxième dense
    biais4 = (float*)malloc(sizeof(float) * 84*1);
    biais4 = Dense2_bias;
    cudaMalloc((void**)&d_biais4, sizeof(float) * 84*1);
    cudaMemcpy(d_biais4, biais4, sizeof(float) * 84*1, cudaMemcpyHostToDevice);
    
    //init(S7_data, d_S7_data, 1, 10, 1, 0, 0, 0);
    S7_data = (float*)malloc(sizeof(float) * 1*10);
    MatrixInit2(S7_data, 1, 10, 0, 0);
    cudaMalloc((void**)&d_S7_data, sizeof(float) * 1*10);
    cudaMemcpy(d_S7_data, S7_data, sizeof(float) * 1*10, cudaMemcpyHostToDevice);       
    
    //init(D3_dense, d_D3_dense, 84, 10, 1, 0, 0, 0);
    D3_dense = (float*)malloc(sizeof(float) * 84*10);
    D3_dense = Dense3_weights;
    cudaMalloc((void**)&d_D3_dense, sizeof(float) * 84*10);
    cudaMemcpy(d_D3_dense, D3_dense, sizeof(float) * 84*10, cudaMemcpyHostToDevice);  
    
    //biais du troisième dense
    biais5 = (float*)malloc(sizeof(float) * 10*1);
    biais5 = Dense3_bias;
    cudaMalloc((void**)&d_biais5, sizeof(float) * 10*1);
    cudaMemcpy(d_biais5, biais5, sizeof(float) * 10*1, cudaMemcpyHostToDevice);    
    
    //init(S8_data, d_S8_data, 1, 10, 1, 0, 0, 0);
    S8_data = (float*)malloc(sizeof(float) * 1*10);
    MatrixInit2(S8_data, 1, 10, 0, 0);
    cudaMalloc((void**)&d_S8_data, sizeof(float) * 1*10);
    cudaMemcpy(d_S8_data, S8_data, sizeof(float) * 1*10, cudaMemcpyHostToDevice);     
    
   
    
    //mise en place des couches du modèle
    
    //C1 : keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same')
    int grid_size = 6;
    dim3 BlockDim1 (28,28);
    cudaConvol<<<grid_size ,BlockDim1>>>(d_raw_data, d_C1_kernel, d_C1_data, 5, 28, d_biais1);

    
    
    //S2 : keras.layers.AveragePooling2D()
    dim3 BlockDim2 (14,14);
    cudaMaxPool<<<grid_size ,BlockDim2>>>(28, 14 , 2 ,d_C1_data , d_S1_data);
    cudaDeviceSynchronize();
    
    
    
    //C3 : keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid')
    //applique les 6 noyaux à chaque image le noyau à 16 listes de 6 kernels
    grid_size = 16;
    dim3 BlockDim3 (10,10);
    cudaConvol2<<<grid_size ,BlockDim3>>>(d_S1_data, d_C3_kernel, d_S3_data, 5, 10, d_biais2);
    
    
    
    //S4 : keras.layers.AveragePooling2D()
    dim3 BlockDim4 (5,5);
    cudaMaxPool<<<grid_size ,BlockDim4>>>(10, 5 , 2 ,d_S3_data , d_S4_data);
    cudaDeviceSynchronize();
    
    
    
    //Flatten : c'est déjà fait !
    
    
    
    //C5 : keras.layers.Dense(120, activation='tanh')
    grid_size = 1;
    int BlockDim = 120;
    cudaMatrixMult<<<grid_size,BlockDim>>>(d_S4_data, d_D1_dense, d_S5_data, 400, d_biais3);
    
    
    
    //F6 : keras.layers.Dense(84, activation='tanh')
    grid_size = 1;
    BlockDim = 84;
    cudaMatrixMult<<<grid_size,BlockDim>>>(d_S5_data, d_D2_dense, d_S6_data, 120, d_biais4);
    
    
    
    //Output layer : keras.layers.Dense(10, activation='softmax')
    grid_size = 1;
    BlockDim = 10;
    cudaMatrixMult2<<<grid_size,BlockDim>>>(d_S6_data, d_D3_dense, d_S7_data, 84, d_biais5);    
    
    
    //retourne sur le CPU pour l'activation softmax
    cudaMemcpy(S7_data, d_S7_data, sizeof(float)*10, cudaMemcpyDeviceToHost);
    cudaMemcpy(S8_data, d_S8_data, sizeof(float)*10, cudaMemcpyDeviceToHost);

        
    activation_softmax(S7_data, S8_data, 10);

    //on regarde quelle est la probabilité maximum : c'est la prédiction
    MatrixPrint2(S8_data, 10, 1, 1);
    
    
      float max;
      int position, longueur=10;

      max=0;
      for (int i=0 ; i< longueur ; i++)
      {
          if(S8_data[i]>max){
                         max=S8_data[i];
                         position=i;
          }
      }
      printf("Estimation : %i\n",position+1);
    
    //ça ne fonctionne parfois mais se trompe très souvent,on autait pu passer en double ou alors on a mal récupéré les poids
   

    
   
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C3_kernel);
    cudaFree(d_S3_data);
    cudaFree(d_S4_data);
    cudaFree(d_S5_data);
    cudaFree(d_D1_dense);
    cudaFree(d_S6_data);
    cudaFree(d_D2_dense);
    cudaFree(d_S7_data);
    cudaFree(d_D3_dense);
    cudaFree(d_S8_data);
    cudaFree(d_biais1);
    cudaFree(d_biais2);
    cudaFree(d_biais3);
    cudaFree(d_biais4);
    cudaFree(d_biais5);
    
  

    free(raw_data);  

    free(C1_data);
        
    free(S1_data);    
   
    //free(C1_kernel);
     
    //free(C3_kernel);
    
    free(S3_data); 
    
    free(S4_data); 
    
    free(S5_data); 
    //free(D1_dense);
    
    free(S6_data); 
    //free(D2_dense);
    free(S7_data); 
    //free(D3_dense); 
    free(S8_data);


}