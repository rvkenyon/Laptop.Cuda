//#define WIN32 1

#include <fstream>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include "stdio.h"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>  

#define THREADS 512
//#define Gy 32 //grid y dimension
//#define Gx 1920 //grid x dimension
//#define Xt 1 //thread x dimension
//#define Yt 64 //thread y dimension
//#define Fx 2000 //number of frames
typedef struct{
	int x;
	int y;
	} twoD;


typedef struct{
	int win;
	float sDev;
	} pixelLoc;

typedef struct{
	int loc_Wind1;
	int loc_Wind2;
	float loc_corrCoef;
	} PixelxCor;

int const Gy = 32; //grid y dimension
int const Gx = 4; //grid x dimension
int const Xt = 512; //thread x dimension
//int const Gx = 1920; //grid x dimension
//int const Xt = 1; //thread x dimension
int const Yt = 1; //32//thread y dimension
int const Fx = 2000; //number of frames
int const h_Wsize = 50;


using namespace std;

//this where each thread takes a different window to Xcorr
//__global__ void XcrossCUDA(int* d_Pixels, pixelLoc* d_PL, float* d_Cor, int N, int corCount, int Wsize)
//{
//	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
//	//	int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float x1, x2, SumPt2, Sum_X1X2, window[120],sdev1,sdev2;
//	int winStart, window1,window2, Index,temp2,temp3; //change yIdx and xIdx
//	//	float xcorrValues[16000];
//
//	// find local point only for xcorr with window
//	if(xIdx < N-1 && d_PL[xIdx].win > 0)//|| window1 > 0) //not at end of file
//	{
//		winStart = xIdx;
//		temp3 = xIdx; //index from "points to correlate" array d_PL
//		window1 = d_PL[winStart].win;
//		//load window for xcorr
//		sdev1 = d_PL[winStart].sDev;
//
//		//temp2 = ((temp3) * (temp3-1))/2;
//		//Index = (N * temp3) - temp2; //this needs to be checked
//		//calculate the offset to write the output data file [N * I - (N(N-1)/2)]
//		//if(xIdx == 0)
//		//	Index = 0;
//		//else 
//		//	{
//		//	temp2 = ;
//		Index = corCount - ((N-xIdx) * (N-xIdx - 1))/2; //this needs to be checked
//		//	}
//
//		//get pixel values for correlation's Master window
//		for(int n = 0; n < Wsize; n++) 
//			window[n] = d_Pixels[window1 + n];
//
//		//now get windows for other points in the correlation
//		for(int i = 0; i < N - xIdx; i++)
//		{
//			window2 = d_PL[winStart+i].win;
//			sdev2 = d_PL[winStart+i].sDev;
//			//			if(sdev2 == 0 || sdev1 == 0) 
//			//			continue;
//			//if(window2 < 0)
//			//	break;
//			//find data start point for windows in silo
//
//			//if point is valid then begin correlations
//			x1 = x2 = Sum_X1X2 = 0.;
//
//			// do the actual cross correlation now
//			for (int l = 0; l < Wsize; l++)
//			{
//				SumPt2 = d_Pixels[window2 + l];
//				x1 += window[l];
//				x2 += SumPt2;
//				Sum_X1X2 += window[l] * SumPt2;
//			}
//			d_Cor[i + Index] = ((Sum_X1X2 - x1 * x2/Wsize)/(Wsize - 1)/sdev2/sdev1);	
//		} //end of correlation calculation
//	} // end of finding windows inside pixel silo
//}

//this where thread acts on the same window to Xcorr
__global__ void XcrossCUDA_same(int* d_Pixels, pixelLoc* d_PL, PixelxCor* d_Cor, int X, int corCount, int Wsize)
	{
	__shared__ int window[h_Wsize];
	//here d_Cor is on Host not Device
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;

	float x1, x2, SumPt2, Sum_X1X2,sdev1,sdev2;
	int winStart, window1,window2, Index,temp2,temp3; //change yIdx and xIdx

	// find local point only for xcorr with window
	if(xIdx < X-1)
		{
		for(int i = 0, j = xIdx; xIdx < X-1 - i;j = xIdx, i++) //increment through all PL data points
			{
			//		if(xIdx < N-i) //not at end of file
			//	{
			Index = corCount - ((X-i) * (X-i - 1))/2; //this needs to be checked
			winStart = i; //index of the window
			window1 = d_PL[winStart].win;
			sdev1 = d_PL[winStart].sDev;
			//get pixel values for correlation's Master window
			if(xIdx < Wsize)
				window[xIdx] = d_Pixels[window1 + xIdx];
			__syncthreads();

			//roll through all the data for this window
			while(j < X-1-i)
				{
				window2 = d_PL[winStart+j].win;
				sdev2 = d_PL[winStart+j].sDev;

				//if point is valid then begin correlations
				x1 = x2 = Sum_X1X2 = 0.;

				// do the actual cross correlation now
				for (int l = 0; l < Wsize; l++)
					{
					SumPt2 = d_Pixels[window2 + l];
					x1 += window[l];
					x2 += SumPt2;
					Sum_X1X2 += window[l] * SumPt2;
					}
				d_Cor[j + Index].loc_corrCoef = ((Sum_X1X2 - x1 * x2/Wsize)/(Wsize - 1)/sdev1/sdev2);	
				d_Cor[j + Index].loc_Wind1 = window1;	
				d_Cor[j + Index].loc_Wind2 = window2;	

				j += gridDim.x * blockDim.x;
				}
			}
		}
	}


__global__ void StdDev(int* d_Pixels, pixelLoc* d_PL,  int Wsize, int frames,  int yTotal, twoD numProcThds, int devThres)
	{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

	float temp, x1=0.f, x2=0.f;
	int xyStart; //where to start reading the window
	int outStart;   //output file indexing
	if(xIdx < numProcThds.x && yIdx < numProcThds.y)
		{
		while(yIdx < yTotal)
			{
			//Calculate Standard Deviation per window within XY pixel silo
			xyStart = xIdx + frames * yIdx;
			outStart = xIdx + numProcThds.x * yIdx;
			//			outStart = xIdx + gridDim.x * blockDim.x * yIdx;
			x1 = x2 = 0.;
			for(int i = 0; i < Wsize; i++)
				{
				temp = d_Pixels[xyStart + i];
				x1 += temp;
				x2 += temp * temp; 
				}
			temp = sqrtf((x2 - x1*x1/Wsize)/(Wsize-1));
			if(temp > devThres)
				{
				d_PL[outStart].sDev = temp;
				d_PL[outStart].win = xyStart;
				}
			else
				{
				d_PL[outStart].win = -1; 
				d_PL[outStart].sDev = 0.0f;
				}
			yIdx += gridDim.y*blockDim.y;
			}
		}
	}


int main()
	{
	int const count = Fx*Gy*Yt; //Fx=MaxX, Gy * Yt = maxY for data file
	twoD numProcThds;
	numProcThds.x = Fx - h_Wsize; 
	numProcThds.y = Gy*Yt;//X*Y*F;
	int const imageX = 172;
	int const imageY = 130;
	int const totalPixs = imageX * imageY;
	int const readSize = Fx * totalPixs;
	int i = 0,N;
	int size_file=0;
	int devThres = 35;
	int procsrTot = numProcThds.x*numProcThds.y;
	pixelLoc *PL; //*h_PL = new pixelLoc[readSize];
	int *Pixels;//*h_Pixels = new int[readSize];
	int *m = new int[readSize];
	int  asd=sizeof(pixelLoc);
	//pixelLoc *d_PL;
	//int *d_Pixels; 
	//PixelxCor *d_Cor;
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) 
		{
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
		}

	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
		{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (long)devProps.totalGlobalMem, 
			(int)devProps.major, (int)devProps.minor, 
			(int)devProps.clockRate);
		}
	int const gridLimit = devProps.maxGridSize[0];
	int const thredMax = devProps.maxThreadsPerBlock;

	const dim3 blockSize(Xt, Yt, 1);  //TODO
	const dim3 gridSize(((Fx-1)/thredMax)+1,Gy, 1);  //TODO
	//int Tot_NumThreads;
	//int BlockWidth;
	//first = Tot_NumThreads/BlockWidth;
	//second = BlockWidth;//threads per block

	int frames = Fx;
	FILE* file;
	file = fopen("c:/data/file_name20.txt", "r");
	if(file == 0)
		{
		printf("bad file name\n");
		exit(0);
		}
	float temp;int ib=0;

		while (!feof (file))
//	for(int i = 0; i < readSize; i++)
		{  
		fscanf(file, "%E", &temp);
		h_Pixels[ib++] = int(temp);
		size_file++;
		}
	int yTotal;
//	readSize = size_file;
	//   std::ofstream fout("c:/data/file_.bin", std::ios::binary);
	//      fout.write(reinterpret_cast<char*>(h_Pixels), sizeof(int) * readSize);
	//fout.close();

	//std::ifstream fin("c:/data/file_.bin", std::ios::binary);
	//fin.read(reinterpret_cast<char*>(h_Pixels), sizeof(int) * readSize);
	//fin.close();

	//for(int i=0; i < readSize; i++)
	//	if(m[i] != h_Pixels[i])
	//		cout<<"wrong"<<endl;

	cout<<"Prior to addition: "<<endl;

	for(int i = 0; i < 10; i++)
		{
		cout<<h_Pixels[i]<<endl;
		}
	temp = 0;

//cudaMalloc((void**) &d_PL, sizeof(pixelLoc) * readSize);
//	cudaMemset((void*) d_PL, 0, sizeof(pixelLoc) * readSize);
//	checkCudaErrors(cudaMalloc((void**) &d_Pixels, sizeof(int) * readSize));

	cudaMallocManaged((void**) &PL, sizeof(pixelLoc) * Fx);
	cudaMemset((void*) d_PL, 0, sizeof(pixelLoc) * Fx);
	cudaMallocManaged((void**) &Pixels, sizeof(int) * Fx);

//	checkCudaErrors(cudaMemcpy((void*) d_Pixels, h_Pixels, sizeof(int) * readSize, cudaMemcpyHostToDevice));

	StdDev<<<gridSize, blockSize>>>(d_Pixels, d_PL, h_Wsize, frames, totalPixs, numProcThds, devThres);

	//rearrange the Loc file for xcorr in next cuda function


	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(h_PL, d_PL, sizeof(pixelLoc) * readSize, cudaMemcpyDeviceToHost));

	//compress list of points, removing points below threshold 
	int j = 0;
	for(int i = 0; i < readSize; i++)
		{
		//if(h_PL[i].sDev < 1 && h_PL[i].win != -1)
		//	{
		//	cout<<"std = "<<h_PL[i].sDev<<"   "<<h_PL[i].win<<endl;
		//	}
		if(h_PL[i].win > 0)
			{
			h_PL[j++] = h_PL[i];
			}
		}
	N = j;
	cudaFree(d_PL);
	cudaFree(d_Pixels);
	checkCudaErrors(cudaMalloc((void**) &d_PL, sizeof(pixelLoc) * N));

	checkCudaErrors(cudaMalloc((void**) &d_Pixels, sizeof(int) * readSize));
	int const N1 = N +1;
	unsigned int const corSize = N1*(N1-1)/2;
	PixelxCor *h_Cor;

	//use memory on Host for Kernel not Device due to Size of Array
	checkCudaErrors(cudaHostAlloc((void**)&h_Cor, sizeof(PixelxCor) * corSize, cudaHostAllocMapped));

	//get the address for Kernel write to output array
	checkCudaErrors(cudaHostGetDevicePointer(&d_Cor, h_Cor, 0));

	//do the regular stuff for passing arrays to Kernel
	checkCudaErrors(cudaMemcpy((void*) d_Pixels, h_Pixels, sizeof(int) * readSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((void*) d_PL, h_PL, sizeof(pixelLoc) * N, cudaMemcpyHostToDevice));

	//int *Indexing = new int[300000];
	//for(int idx = 0; idx < N; idx++)
	//	Indexing[idx] = corSize - ((N1-idx) * (N1-idx - 1))/2; //this needs to be checked

	//now do xcorrelation
	int  blocks = (N+thredMax-1)/thredMax;
	if(blocks > gridLimit) blocks = gridLimit;

	XcrossCUDA_same<<<blocks, thredMax>>>(d_Pixels, d_PL,  d_Cor, N1, corSize, h_Wsize);

	checkCudaErrors(cudaDeviceSynchronize()); 
	checkCudaErrors(cudaGetLastError());
	delete[] h_Pixels;
	cudaFree(d_Pixels);

	//	checkCudaErrors(cudaMemcpy(h_Cor, d_Cor, sizeof(float) * corSize, cudaMemcpyDeviceToHost));

	//	int ja = 0;
	//	float *pp = new float[300000];
	//	for(int i = 0, temp = 0; i < corSize; i++)
	//	{
	//		temp = 1000.0*h_Cor[i];
	//		if(abs((int) temp) > 998)
	//		{
	//			pp[ja++] = i;
	////			ja++;
	//		}
	//	}
	//	cout<<"After addition:"<<ja<<endl;
	////create file for Lovain analysis: P1 X Pn....; P2 X Pn-1....
	//for(int i = 0; i < corSize; i++)
	//{
	//	h_final[i].addrss = h_PL[i].win;
	//	h_final[i].Xaddrss = h_PL[i + j].win;
	//	h_final[i].XcorVal = h_Cor[j];
	//}
	//	cout<<ja<<endl;


	int Xloc1, Yloc1, Floc1;
	int Xloc2, Yloc2, Floc2;



		//write out the data to a file
		FILE *fpw;
		char filew[512];
		sprintf(filew,"%s.pair.txt","cor_weights");
		if ((fpw = fopen(filew,"w"))==NULL)
			{
			printf("cannot open file\n");
			}

		fprintf(fpw, "Pixel No\tNode 1\tNode 2\tScale\n");
		for(int i = 0; i < corSize; i++)
			{
			Floc1 = h_Cor[i].loc_Wind1 % frames;
			Floc2 = h_Cor[i].loc_Wind2 % frames;
			Xloc1 = floor((h_Cor[i].loc_Wind1/imageY));
			Yloc1 = (h_Cor[i].loc_Wind1-Floc1) - (Xloc1*imageY);
			Xloc1 = Xloc1 + 1;
			if (~Yloc1)
				Yloc1=imageX;
			Xloc2 = floor((h_Cor[i].loc_Wind2-Floc2)/imageY);
			Yloc2 = (h_Cor[i].loc_Wind2-Floc2) - (Xloc2*imageY);
			if (~Yloc2)
				Yloc2=imageX;
			fprintf(fpw, "%d\t%d\t%d\t%f\n",Xloc2, Yloc2, Floc2, h_Cor[i].loc_corrCoef);
			}

		fclose(fpw);
		cudaFreeHost(h_Cor);
		cudaFree(d_PL);
		delete[] h_PL;
		return 0;
	}
