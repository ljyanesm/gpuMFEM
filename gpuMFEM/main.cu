//This is the simplest possible implementation of the paper
//"Point based animation of elastic, plastic and melting objects" by Matthias Mueller et. al"
//(http://www.matthiasmueller.info/publications/sca04.pdf) and the detailed chapter on
//Meshless Finite Elements, Chapter 7 in Point-Based Graphics,Markus Gross,
//Hanspeter Pfister (eds.), ISBN 0123706041,pp: 341--357, 2007.

//Controls:
//left click on any empty region to rotate, middle click to zoom
//left click and drag any point to drag it.
//Press '[' or ']' to decrease/increase Poisson Ratio (nu)
//Press ',' or '.' to decrease/increase Young's Modulus (Y)
//Press 'j' to view the Jacobians
//Press 'f' to view the Forces which maybe
//      'x' for displacement,
//      'e' for strains and
//      's' for stresses

// This example is based on 
// https://code.google.com/p/opencloth/source/browse/trunk/OpenCloth_MeshlessFEM/OpenCloth_MeshlessFEM/main.cpp?r=54

#define _USE_MATH_DEFINES

#include <omp.h>

#include <cmath>
#define M_PI       3.14159265358979323846

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <glm/glm.hpp>

using namespace std;

typedef unsigned int uint;

struct Neighboor
{
	glm::vec4	rdist;
	glm::vec4	dj;
	int			j;
	float		w;
};

class DeformableModel
{
public:

	enum ParallelismType {
		NONE,
		OPENMP,
		CUDA
	};

	enum FileFormat{
		VTK,
		CSV
	};

private:

	typedef std::pair<int, float> mypair;

	std::string simulationName;
	int numParticles;

	std::vector<glm::mat3>	J;
	std::vector<glm::mat3>	sigma;
	std::vector<glm::mat3>	epsilon;
	std::vector<glm::mat3>	Minv;
	std::vector<glm::vec4>	P;
	std::vector<glm::vec4>	Pi;
	std::vector<glm::vec4>	di;
	std::vector<glm::vec4>	V;
	std::vector<glm::vec4>	Acc;
	std::vector<glm::vec4>	F;
	std::vector<glm::vec4>	U;
	std::vector<float>		M;
	std::vector<float>		r;
	std::vector<float>		h;
	std::vector<float>		rho;
	std::vector<float>		Vol;
	std::vector<int>		isFixed;

	glm::mat3		*dev_J;
	glm::mat3		*dev_sigma;
	glm::mat3		*dev_epsilon;
	glm::mat3		*dev_Minv;
	glm::vec4		*dev_P;
	glm::vec4		*dev_Pi;
	glm::vec4		*dev_di;
	glm::vec4		*dev_V;
	glm::vec4		*dev_Acc;
	glm::vec4		*dev_F;
	glm::vec4		*dev_U;
	float			*dev_M;
	float			*dev_r;
	float			*dev_h;
	float			*dev_rho;
	float			*dev_Vol;
	int				*dev_isFixed;
	Neighboor		*dev_neighboors;

	float					Young_modulus;
	float					Poisson_ratio;

	float					d15, d16, d17, d18;
	glm::vec3				D;
	std::vector<Neighboor> neighboors;

	int		neighSize;
	float	gravity;
	float	elasticityKernelConstant;

	float	velocityDamping;
	float	kv;

	float	modelDensity;
	float	massScaling;

	float	deltaTime;
	float	simulationTime;
	float	collisionDamping;

	bool	paused;

	int		singularMatrixCnt;
	int		numIterations;

	bool	logSimulation;
	int		outputIterations;

	ParallelismType currentParallelism;
public:

	DeformableModel(vector<glm::vec4> mesh, float Young_Modulus, float Poisson_ratio, bool log_);
	~DeformableModel(void);

	void stepPhysics(float dt, ParallelismType type, bool forceStep = false);
	void togglePaused(){ paused = !paused;}
	void start(){ paused = false; }
	void stop() { paused = true;  }

	void _uploadDeviceData();
	void _uploadPosVel();
	std::vector<glm::vec4> const *getPositions(ParallelismType type);
	std::vector<glm::vec4> const *getVelocities(ParallelismType type);
	std::vector<Neighboor> const *getNeighboors();
private:

	struct Cmp {
		bool operator()(const mypair &lhs, const mypair &rhs) {
			return lhs.second < rhs.second;
		}
	};
	void _outputSystem(FileFormat format = VTK);

	void _calculateNeighboors();

	void _calculateDensities();
	void _calculateDeformation();
	void _calculateForces();
	void ComputeJacobians();
	void ComputeStrainAndStress();
	void _integrateLeapFrog( float dt);

	void _initializeMass();

	float elasticityKernel(float r);
	float elasticityKernelDerivative(float r);
	glm::vec4 elasticityKernelDerivative(glm::vec4 r);

	void _calculateConstants();
	void getKNearestNeighboors( int i, int k, std::vector<float> &dist, std::vector<Neighboor> &neighboors );
	void computeRadiusAndSupportRadii( std::vector<float> &dist, float &r, float &h );
	void fillNeighboors( int i );
	void computeScalingConstant();
	void ComputeMass( float modelDensity, int i );
	void ComputeDensityAndVolume( int i );
	void ComputeInvMomentMatrix( int i );

	void calculateForcesCUDA();
	void ComputeJacobiansCUDA();
	void ComputeStrainAndStressCUDA();
	void integrateLeapFrogCUDA( float dt );
	void _vtkOutput(std::ostringstream &filename);
	void _csvOutput(std::ostringstream &filename);
	void _donwloadDeviceData();
};

// Compute number of blocks to create
int iDivUp (int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int maxThreads, int &numBlocks, int &numThreads)
{
	numThreads = min( maxThreads, numPnts );
	numBlocks = iDivUp ( numPnts, numThreads );
}

__global__ void dev_applyGravityDamping(glm::vec4 *F, glm::vec4 *V, int *isFixed, float gravity, float velocityDamping, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;

	if(!isFixed[i])
		F[i] = glm::vec4(0,0,gravity, 0);
	else
		F[i] = glm::vec4(0);

	//Add velocity damping
	F[i] -= V[i] * velocityDamping;
}

__global__ void dev_updateVelocity_collisions(int *isFixed, glm::vec4 *P, glm::vec4 *V, glm::vec4 *Acc, glm::vec4 *F, float *M, float dt, float half_dt2, float collisionDamping, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;

	if(!isFixed[i]) {
		Acc[i] = F[i]/M[i];

		P[i] += dt*V[i]+(Acc[i]*half_dt2);

	}
}

__global__ void dev_leapfrogVelocities(int *isFixed, glm::vec4 *V, glm::vec4 *F, float *M, glm::vec4 *Acc, float half_dt2, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;

	if(!isFixed[i])
		V[i] += ((F[i]/M[i] + Acc[i])*half_dt2);
}

////Calculate internal force using the stress and Jacobians
__global__ void dev_CalculateInternalForce(float *Vol, glm::mat3 *J, glm::mat3 *sigma, glm::vec4 *di, glm::vec4 *F, Neighboor *neighboors, float kv, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;

	uint index = i*10;

	glm::mat3 F_e;
	glm::mat3 F_v;
	F_e =  -2.0f * Vol[i] * (glm::mat3(J[i]) * glm::mat3(sigma[i])) ;        // Eq. 18
	glm::vec3 J_u = glm::vec3(J[i][0][0], J[i][0][1],J[i][0][2]);
	glm::vec3 J_v = glm::vec3(J[i][1][0], J[i][1][1],J[i][1][2]);
	glm::vec3 J_w = glm::vec3(J[i][2][0], J[i][2][1],J[i][2][2]);

	glm::vec3 row0 = glm::cross(J_v, J_w);  //Eq. 22
	glm::vec3 row1 = glm::cross(J_w, J_u);  //Eq. 22
	glm::vec3 row2 = glm::cross(J_u, J_v);  //Eq. 22
	glm::mat3 M= glm::transpose(glm::mat3(row0, row1, row2));       //Eq. 22

	F_v = -Vol[i] * kv * (glm::determinant(glm::mat3(J[i])) - 1) * M ; //Eq. 22

	for(size_t j=0;j<10;j++){
		uint nJ = neighboors[index+j].j;
		glm::vec4 nDJ = neighboors[index+j].dj;
		F[nJ] += glm::vec4((F_e + F_v) * glm::vec3(nDJ), 0);
	}

	F[i] += glm::vec4((F_e + F_v) * glm::vec3(di[i]), 0);

}

__global__ void dev_ComputeJacobians(Neighboor *neighboors, glm::vec4 *U, glm::vec4 *P, glm::vec4 *Pi, glm::mat3 *J, glm::mat3 *Minv, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;
	uint index = i*10;
	for(int j=0;j<10;j++){
		uint nJ = neighboors[index+j].j;
		U[nJ] = P[nJ] - Pi[nJ];
	}

	glm::mat3 B=glm::mat3(0);               // help matrix used to compute the sum in Eq. 15

	// reset du and du_tr
	glm::mat3 du=glm::mat3(0);
	glm::mat3 du_tr=glm::mat3(0);

	for(int j=0;j<10;j++)
	{
		uint nJ = neighboors[index+j].j;
		glm::vec4 nJ_rdist = neighboors[index+j].rdist;
		float nJ_w = neighboors[index+j].w;
		glm::mat3 Bj=glm::mat3(0);
		//Eq. 15 right hand side terms with A_inv
		Bj = glm::outerProduct(glm::vec3(U[nJ] - U[i]), glm::vec3( nJ_rdist * nJ_w) );
		B += Bj;
	}
	B = glm::transpose(B);

	du = Minv[i] * B;       // Eq. 15 page 4
	du_tr = glm::transpose(du);
	J[i]=glm::mat3(1);
	J[i] += du_tr;          // Eq. 1
}

__global__ void dev_ComputeStrainAndStress(glm::mat3 *J, glm::mat3 *epsilon, glm::mat3 *sigma, glm::vec3 D, int numParticles)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numParticles ) return;

	glm::mat3 Jtr = glm::transpose(glm::mat3(J[i]));
	epsilon[i] = (Jtr * J[i]) - glm::mat3(1);          // formula 3, Green-Saint Venant non-linear tensor

	glm::mat3& e= epsilon[i];
	glm::mat3& s= sigma[i];

	s[0][0] = D.x*e[0][0]+D.y*e[1][1]+D.y*e[2][2];
	s[1][1] = D.y*e[0][0]+D.x*e[1][1]+D.y*e[2][2];
	s[2][2] = D.y*e[0][0]+D.y*e[1][1]+D.x*e[2][2];

	s[0][1] = D.z*e[0][1];
	s[1][2] = D.z*e[1][2];
	s[2][0] = D.z*e[2][0];

	s[0][2] = s[2][0];
	s[1][0] = s[0][1];
	s[2][1] = s[1][2];
}
DeformableModel::DeformableModel(vector<glm::vec4> positions, float Y, float v, bool log_) :
Young_modulus(Y),
	Poisson_ratio(v),
	singularMatrixCnt(0),
	simulationTime(0),
	numIterations(0),
	logSimulation(log_),
	paused(true)
{
	simulationName = std::string("meshlessFEM");
	outputIterations = 1000;
	neighSize = 10;
	// Reserve all the memory needed
	numParticles = positions.size();

	P.resize(numParticles);
	Pi.resize(numParticles);
	isFixed.resize(numParticles);
	V.resize(numParticles);
	F.resize(numParticles);
	U.resize(numParticles);
	M.resize(numParticles);
	r.resize(numParticles);
	h.resize(numParticles);
	Vol.resize(numParticles);
	rho.resize(numParticles);
	di.resize(numParticles);
	J.resize(numParticles);
	sigma.resize(numParticles);
	epsilon.resize(numParticles);
	Acc.resize(numParticles);
	Minv.resize(numParticles);
	neighboors.resize(numParticles*neighSize);

	cudaMalloc((void **) &dev_P,	numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_Pi,	numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_V,	numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_F,	numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_U,	numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_M,	numParticles*sizeof(float));
	cudaMalloc((void **) &dev_r,	numParticles*sizeof(float));
	cudaMalloc((void **) &dev_h,	numParticles*sizeof(float));
	cudaMalloc((void **) &dev_isFixed, numParticles*sizeof(int));
	cudaMalloc((void **) &dev_Vol, numParticles*sizeof(float));
	cudaMalloc((void **) &dev_rho, numParticles*sizeof(float));
	cudaMalloc((void **) &dev_di, numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_J,	numParticles*sizeof(glm::mat3));
	cudaMalloc((void **) &dev_sigma,	numParticles*sizeof(glm::mat3));
	cudaMalloc((void **) &dev_epsilon, numParticles*sizeof(glm::mat3));
	cudaMalloc((void **) &dev_Acc, numParticles*sizeof(glm::vec4));
	cudaMalloc((void **) &dev_Minv, numParticles*sizeof(glm::mat3));
	cudaMalloc((void **) &dev_neighboors, numParticles*neighSize*sizeof(Neighboor));

	std::fill(isFixed.begin(), isFixed.end(), false);
	P = positions;
	Pi = P;
	cudaMemcpy((void *) dev_Pi, &P[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	for (int i = 0; i < numParticles; i++)
	{
		V[i] = glm::vec4(0,0,0,0);
	}

	// Initialize all the variables
	collisionDamping = 0.8f;

	gravity = -9.81f;
	modelDensity = 1000.0f;
	velocityDamping = 500.0f;
	kv = 100.0f;

	d15 = Young_modulus / (1.0f + Poisson_ratio) / (1.0f - 2 * Poisson_ratio);
	d16 = (1.0f - Poisson_ratio) * d15;
	d17 = Poisson_ratio * d15;
	d18 = Young_modulus / 2 / (1.0f + Poisson_ratio);

	D = glm::vec3(d16, d17, d18); //Isotropic elasticity matrix D
	//(in the original paper this matrix is called C) in Eq. 4 on page 3

	for (int i = 0; i < numParticles; i++) {
		std::vector<float> dist;
		getKNearestNeighboors(i, neighSize, dist, neighboors);
		computeRadiusAndSupportRadii(dist, r[i], h[i]);
	}

	for (int i = 0; i < numParticles; i++){
		fillNeighboors(i);
	}

	computeScalingConstant();

	for(int i = 0; i < numParticles; ++i)
		ComputeMass(modelDensity, i);

	for(int i = 0; i < numParticles; ++i)
		ComputeDensityAndVolume(i);

	for(int i = 0; i < numParticles; ++i) {
		ComputeInvMomentMatrix(i);
	}

	_uploadDeviceData();
}

// Initialization step
void DeformableModel::getKNearestNeighboors( int index, int k, std::vector<float> &dist, std::vector<Neighboor> &neighboors ) {
	std::vector<mypair> distances;

	//Get the distances of current point to all other points
	for(int i = 0; i < numParticles; i++) {
		if(index!=i) {
			mypair m;
			m.first = i;
			m.second= fabs(glm::distance(P[index], P[i]));
			distances.push_back(m);
		}
	}

	//sort the distances
	sort (distances.begin(), distances.end(), Cmp());

	//now take the top k neighbors based on distance
	for(int i=0;i<k;i++) {
		Neighboor ne;
		ne.j = distances[i].first;
		dist.push_back(distances[i].second);
		neighboors[(index*neighSize) + i] = ne;
	}
}

// Initialization step
void DeformableModel::computeRadiusAndSupportRadii( std::vector<float> &dist, float &r, float &h )
{
	//For this read section 3.2 Initialization on page 4
	//look through all neighbor distances
	//and average the distance. This will give us r
	float avg = 0.f;
	for(size_t i=0;i<dist.size();i++)
		avg += dist[i];
	r = avg / dist.size();

	// compute the support radii H = 3 * R
	h = 3.0f * r;
}

// Initialization step
void DeformableModel::fillNeighboors( int index )
{
	//For this read section 3.2 Initialization on page 4
	//Based on Eq. 9
	float mul_factor = float(315.0f/(64*M_PI*pow(h[index],9.0f)));//35 / (float)(32 * M_PI * pow(h, 7));
	float h2 = h[index]*h[index];
	for(int i = 0; i < neighSize; i++)
	{
		Neighboor& n = neighboors[(index*neighSize) +i];
		n.rdist  = Pi[n.j] - Pi[index];
		float r2 = glm::dot(n.rdist, n.rdist);          //r = sqrt(Xij.x*Xij.x+Xij.y*Xij.y+Xij.z*Xij.z)
		//r^2 = dot(Xij,Xij);
		n.w = mul_factor * pow(h2 - r2, 3 );
	}
}

// Initialization step
void DeformableModel::computeScalingConstant() {
	printf("ComputeScalingFactor: estimating the scaling factor...\n");
	massScaling = 0.f;
	int i=0;
	//#pragma omp parallel for
	for ( i =0; i<numParticles; i++ ) {
		float sum = 0.f;

		for(int j=0;j<neighSize;j++) {
			sum += pow(r[neighboors[(i*neighSize) + j].j], 3) * neighboors[(i*neighSize) + j].w;
		}
		massScaling += 1.0f / sum;
	}
	// This is the common scaling factor to compute the mass of each phyxel.
	// See last paragraph of Section 3.2 on page 4
	massScaling /= numParticles;

	printf("Scaling factor: %3.3f\n", massScaling);
}

// Initialization step
void DeformableModel::ComputeMass( float modelDensity, int index )
{
	// See last paragraph of Section 3.2 on page 4
	M[index] = massScaling * pow(r[index], 3) * modelDensity;
}

// Initialization step
void DeformableModel::ComputeDensityAndVolume( int index )
{
	// See last paragraph of Section 3.2 on page 4
	rho[index] = 0.f;

	for(int j=0;j<neighSize;j++)
		rho[index] += M[neighboors[(index*neighSize) + j].j] * neighboors[(index*neighSize) + j].w;               // Eq. 10 page 4
	Vol[index] =  M[index] / rho[index];
}

// Initialization step
void DeformableModel::ComputeInvMomentMatrix( int index )
{
	glm::mat3 A, A_sum, V;
	A =glm::mat3(0);

	for(int j=0;j<neighSize;j++)
	{
		A_sum = glm::outerProduct(glm::vec3(neighboors[(index*neighSize) + j].rdist), 
			glm::vec3(neighboors[(index*neighSize) + j].rdist) * neighboors[(index*neighSize) + j].w);     // Eq. 14
		A += A_sum;
	}

	if(glm::determinant(A) != 0.0)
		Minv[index] = glm::inverse(A);          // Eq. 14, inverted moment matrix
	else
	{
		// if MomentMatrix is not invertible it means that there are less than 4 neighbors
		// or the neighbor phyxels are coplanar or colinear
		// We should use SVD to extract the inverse but I have left this as is.
		//Read section 3.3 last paragraph



		printf("Warning: Singular matrix! #%d\n", ++singularMatrixCnt);
	}

	di[index]=glm::vec4(0);

	for(int j=0;j<neighSize;j++)
	{
		glm::vec4 Xij_Wij = neighboors[(index*neighSize) + j].rdist * neighboors[(index*neighSize) + j].w;
		neighboors[(index*neighSize) + j].dj = glm::vec4(Minv[index] * glm::vec3(Xij_Wij), 0);        //Eq. 21 page 5
		di[index] -= neighboors[(index*neighSize) + j].dj;                           //Eq. 20 page 5
	}
}

DeformableModel::~DeformableModel(void)
{
	P.clear();
	Pi.clear();
	isFixed.clear();
	V.clear();
	F.clear();
	U.clear();
	M.clear();
	r.clear();
	h.clear();
	Vol.clear();
	rho.clear();
	di.clear();
	J.clear();
	sigma.clear();
	epsilon.clear();
	Acc.clear();
	Minv.clear();
	neighboors.clear();

	cudaFree(dev_P);
	cudaFree(dev_P);
	cudaFree(dev_Pi);
	cudaFree(dev_isFixed);
	cudaFree(dev_V);
	cudaFree(dev_F);
	cudaFree(dev_U);
	cudaFree(dev_M);
	cudaFree(dev_r);
	cudaFree(dev_h);
	cudaFree(dev_Vol);
	cudaFree(dev_rho);
	cudaFree(dev_di);
	cudaFree(dev_J);
	cudaFree(dev_sigma);
	cudaFree(dev_epsilon);
	cudaFree(dev_Acc);
	cudaFree(dev_Minv);
	cudaFree(dev_neighboors);
}

void DeformableModel::stepPhysics( float dt, ParallelismType type, bool forceStep /*= false*/ )
{
	if (paused && !forceStep) return;
	simulationTime += dt;
	numIterations++;

	if (type != currentParallelism && type == CUDA) _uploadDeviceData();

	if (type != currentParallelism && type != CUDA) _donwloadDeviceData();

	currentParallelism = type;

	switch (type)
	{
	case NONE:
		{
			_calculateForces();
			_integrateLeapFrog(dt);
		}
		break;

	case OPENMP:
		{
#pragma omp parallel firstprivate(dt)
			{
				_calculateForces();
				_integrateLeapFrog(dt);
			}
		}
		break;

	case CUDA:
		calculateForcesCUDA();
		integrateLeapFrogCUDA(dt);
		break;
	}

	if (logSimulation && numIterations % outputIterations == 0)
	{
		_outputSystem(VTK);
	}
}

void DeformableModel::_calculateForces()
{
#pragma omp for
	for(int i=0;i<numParticles;i++) {
		if(!isFixed[i])
			F[i] = glm::vec4(0,0,gravity, 0);
		else
			F[i] = glm::vec4(0);

		//Add velocity damping
		F[i] -= V[i] * velocityDamping;
	}

	ComputeJacobians();

	ComputeStrainAndStress();


	//Calculate internal force using the stress and Jacobians
#pragma omp for
	for(int i=0;i<numParticles;i++) {
		glm::mat3 F_e;
		glm::mat3 F_v;
		F_e =  -2 * Vol[i] * (glm::mat3(J[i]) * glm::mat3(sigma[i])) ;        // Eq. 18
		glm::vec3 J_u = glm::vec3(J[i][0][0], J[i][0][1],J[i][0][2]);
		glm::vec3 J_v = glm::vec3(J[i][1][0], J[i][1][1],J[i][1][2]);
		glm::vec3 J_w = glm::vec3(J[i][2][0], J[i][2][1],J[i][2][2]);

		glm::vec3 row0 = glm::cross(J_v, J_w);  //Eq. 22
		glm::vec3 row1 = glm::cross(J_w, J_u);  //Eq. 22
		glm::vec3 row2 = glm::cross(J_u, J_v);  //Eq. 22
		glm::mat3 M= glm::transpose(glm::mat3(row0, row1, row2));       //Eq. 22

		F_v = -Vol[i] * kv * (glm::determinant(glm::mat3(J[i])) - 1) * M ; //Eq. 22

		for(int j=0;j<neighSize;j++)
			F[neighboors[(i*neighSize) + j].j] += glm::vec4((F_e + F_v) * glm::vec3(neighboors[(i*neighSize) + j].dj), 0);

		F[i] += glm::vec4((F_e + F_v) * glm::vec3(di[i]), 0);
	}
}

void DeformableModel::ComputeJacobians()
{
#pragma omp for
	for(int i=0;i<numParticles;i++) {
		for(int j=0;j<neighSize;j++)
			U[neighboors[(i*neighSize) + j].j] = P[neighboors[(i*neighSize) + j].j] - Pi[neighboors[(i*neighSize) + j].j];

		glm::mat3 B=glm::mat3(0);               // help matrix used to compute the sum in Eq. 15

		// reset du and du_tr
		glm::mat3 du=glm::mat3(0);
		glm::mat3 du_tr=glm::mat3(0);

		for(int j=0;j<neighSize;j++)
		{
			glm::mat3 Bj=glm::mat3(0);
			//Eq. 15 right hand side terms with A_inv
			Bj = glm::outerProduct(glm::vec3(U[neighboors[(i*neighSize) + j].j] - U[i]), 
				glm::vec3(neighboors[(i*neighSize) + j].rdist * neighboors[(i*neighSize) + j].w) );
			B += Bj;
		}
		B = glm::transpose(B);

		du = Minv[i] * B;       // Eq. 15 page 4
		du_tr = glm::transpose(du);
		J[i]=glm::mat3(1);
		J[i] += du_tr;          // Eq. 1
	}
}

void DeformableModel::ComputeStrainAndStress()
{

#pragma omp for
	for(int i=0;i<numParticles;i++) {
		glm::mat3 Jtr = glm::transpose(glm::mat3(J[i]));
		epsilon[i] = (Jtr * J[i]) - glm::mat3(1);          // formula 3, Green-Saint Venant non-linear tensor

		glm::mat3& e= epsilon[i];
		glm::mat3& s= sigma[i];

		s[0][0] = D.x*e[0][0]+D.y*e[1][1]+D.y*e[2][2];
		s[1][1] = D.y*e[0][0]+D.x*e[1][1]+D.y*e[2][2];
		s[2][2] = D.y*e[0][0]+D.y*e[1][1]+D.x*e[2][2];

		s[0][1] = D.z*e[0][1];
		s[1][2] = D.z*e[1][2];
		s[2][0] = D.z*e[2][0];

		s[0][2] = s[2][0];
		s[1][0] = s[0][1];
		s[2][1] = s[1][2];
	}
}

void DeformableModel::_integrateLeapFrog( float dt )
{
	float dt2 = dt*dt;
	float half_dt2 = dt2*0.5f;

#pragma omp for
	for(int i=0;i<numParticles;i++) {

		if(!isFixed[i]) {
			Acc[i] = F[i]/M[i];

			P[i] += dt*V[i]+(Acc[i]*half_dt2);

		}
	}
	//Calculate the new acceleration
	_calculateForces();

#pragma omp for
	for(int i=0;i<numParticles;i++) {
		if(!isFixed[i])
			V[i] += ((F[i]/M[i] + Acc[i])*half_dt2);
	}
}

void DeformableModel::calculateForcesCUDA()
{
	int numBlocks;
	int numThreads;

	computeNumBlocks(numParticles, 128, numBlocks, numThreads);

	dev_applyGravityDamping<<< numBlocks, numThreads>>>(
		dev_F, 
		dev_V, 
		dev_isFixed,
		gravity,
		velocityDamping,
		numParticles);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_applyGravityDamping: %s\n", cudaGetErrorString(error) );
	}   

	ComputeJacobiansCUDA();

	ComputeStrainAndStressCUDA();

	dev_CalculateInternalForce<<< numBlocks, numThreads>>>(
		dev_Vol, 
		dev_J, 
		dev_sigma, 
		dev_di,
		dev_F,
		dev_neighboors,
		kv,
		numParticles);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_CalculateInternalForce: %s\n", cudaGetErrorString(error) );
	}   
}

void DeformableModel::ComputeJacobiansCUDA()
{
	int numBlocks;
	int numThreads;

	computeNumBlocks(numParticles, 128, numBlocks, numThreads);

	dev_ComputeJacobians<<< numBlocks, numThreads>>>(
		dev_neighboors,
		dev_U,
		dev_P,
		dev_Pi,
		dev_J,
		dev_Minv,
		numParticles);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_ComputeJacobians: %s\n", cudaGetErrorString(error) );
	}

}

void DeformableModel::ComputeStrainAndStressCUDA()
{
	int numBlocks;
	int numThreads;

	computeNumBlocks(numParticles, 128, numBlocks, numThreads);

	dev_ComputeStrainAndStress<<< numBlocks, numThreads>>>(
		dev_J,
		dev_epsilon,
		dev_sigma,
		D,
		numParticles);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_ComputeStrainAndStress: %s\n", cudaGetErrorString(error) );
	}   

}

void DeformableModel::integrateLeapFrogCUDA( float dt )
{
	int numBlocks;
	int numThreads;

	computeNumBlocks(numParticles, 128, numBlocks, numThreads);

	float dt2 = dt*dt;
	float half_dt2 = dt2*0.5f;

	dev_updateVelocity_collisions<<< numBlocks, numThreads>>>(
		dev_isFixed,
		dev_P,
		dev_V,
		dev_Acc,
		dev_F,
		dev_M,
		dt,
		half_dt2,
		collisionDamping,
		numParticles);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_updateVelocity_collisions: %s\n", cudaGetErrorString(error) );
	}   

	calculateForcesCUDA();

	dev_leapfrogVelocities<<< numBlocks, numThreads>>>(
		dev_isFixed,
		dev_V,
		dev_F,
		dev_M,
		dev_Acc,
		half_dt2,
		numParticles);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf ( stderr, "CUDA ERROR: dev_leapfrogVelocities: %s\n", cudaGetErrorString(error) );
	}   

}

void DeformableModel::_outputSystem( FileFormat format )
{
	std::ostringstream filename;
	filename << ".\\out\\" << simulationName << "_" << numIterations;

	switch (format)
	{
	case VTK:
		if (currentParallelism == CUDA) _donwloadDeviceData();
		_vtkOutput(filename);
		break;
	case CSV:
		_csvOutput(filename);
		break;
	}

	std::cout << "File: " << filename.str() << " created successfully" << std::endl;
}

void DeformableModel::_vtkOutput( std::ostringstream &filename )
{
	filename << ".vtu";
	ofstream output_file;
	output_file.open(filename.str().c_str(), ofstream::out);

	output_file << "<?xml version=\"1.0\"?>" << endl;
	output_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\">" << endl;
	output_file << "<UnstructuredGrid>" << endl;
	output_file << "<Piece NumberOfPoints=\"" << numParticles << "\" NumberOfCells=\"0\">" << endl;

	// Output positions
	output_file << "<Points>" << endl;

	output_file << "<DataArray NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << P[i].x << " " << P[i].y << " " << P[i].z << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;
	output_file << "</Points>" << endl;

	// Create the useless cell stuff
	output_file << "<Cells>" << endl;
	output_file << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;
	output_file << "0" << endl;
	output_file << "</DataArray>" << endl;
	output_file << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;
	output_file << "0" << endl;
	output_file << "</DataArray>" << endl;
	output_file << "<DataArray type=\"Uint8\" Name=\"types\" format=\"ascii\">" << endl;
	output_file << "1" << endl;
	output_file << "</DataArray>" << endl;
	output_file << "</Cells>" << endl;

	output_file << "<PointData Scalars=\"radius\">" << endl;

	output_file << "<DataArray Name=\"Points\" NumberOfComponents=\"3\" type=\"Float32\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << P[i].x << " " << P[i].y << " " << P[i].z << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	// Output radius
	output_file << "<DataArray type=\"Float32\" Name=\"radius\" NumberOfComponents=\"1\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << h[i] << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	// Output mass
	output_file << "<DataArray type=\"Float32\" Name=\"mass\" NumberOfComponents=\"1\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << M[i] << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	// Output velocity
	output_file << "<DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << V[i].x << " " << V[i].y << " " << V[i].z << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	// Output force
	output_file << "<DataArray type=\"Float32\" Name=\"force\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << F[i].x << " " << F[i].y << " " << F[i].z << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	// Output deformation (U)
	output_file << "<DataArray type=\"Float32\" Name=\"deformation\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;
	for (int i = 0; i < numParticles; i++) {
		output_file << U[i].x << " " << U[i].y << " " << U[i].z << " ";
	}
	output_file << endl;
	output_file << "</DataArray>" << endl;

	output_file << "</PointData>" << endl;

	output_file <<	"</Piece>" << endl;
	output_file << "</UnstructuredGrid>" << endl;
	output_file <<	"</VTKFile>" << endl;

	output_file.close();
}

void DeformableModel::_csvOutput( std::ostringstream &filename )
{
	filename << ".csv";
	std::ofstream output_file;
	output_file.open(filename.str().c_str(), std::ofstream::out);
	output_file << "x, y, z\n";
	for (int i = 0; i < numParticles; i++)
	{
		output_file << P[i].x << ", " << P[i].y << ", " << P[i].z << std::endl;
	}
	output_file.close();
}

std::vector<glm::vec4> const * DeformableModel::getPositions( ParallelismType type )
{
	if (type == CUDA){
		cudaThreadSynchronize ();
		cudaMemcpy((void *) &P[0], (void*) dev_P, numParticles*sizeof(glm::vec4), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize ();
	}

	return &P;
}

std::vector<glm::vec4> const * DeformableModel::getVelocities( ParallelismType type )
{
	if (type == CUDA){
		cudaThreadSynchronize ();
		cudaMemcpy((void *) &V[0], (void*) dev_V, numParticles*sizeof(glm::vec4), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize ();
	}

	return &V;
}

std::vector<Neighboor> const * DeformableModel::getNeighboors()
{
	return &neighboors;
}

void DeformableModel::_uploadDeviceData()
{
	/*
	*	Copy all the information to the GPU for GPU stepping of the model!
	*/
	cudaMemcpy((void *) dev_P, &P[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_V, &V[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_F, &F[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_U, &U[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_M, &M[0], numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_r, &r[0], numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_h, &h[0], numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_isFixed, &isFixed[0], numParticles*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_Vol, &Vol[0], numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_rho, &rho[0], numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_di, &di[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_J, &J[0], numParticles*sizeof(glm::mat3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_sigma, &sigma[0], numParticles*sizeof(glm::mat3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_epsilon, &epsilon[0], numParticles*sizeof(glm::mat3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_Acc, &Acc[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_Minv, &Minv[0], numParticles*sizeof(glm::mat3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_neighboors, &neighboors[0], numParticles*neighSize*sizeof(Neighboor), cudaMemcpyHostToDevice);
}

void DeformableModel::_uploadPosVel()
{
	/*
	*	Copy all the information to the GPU for GPU stepping of the model!
	*/
	cudaMemcpy((void *) dev_P, &P[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) dev_V, &V[0], numParticles*sizeof(glm::vec4), cudaMemcpyHostToDevice);
}


void DeformableModel::_donwloadDeviceData()
{
	cudaMemcpy((void *) &P[0],			dev_P,			numParticles*sizeof(glm::vec4),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &V[0],			dev_V,			numParticles*sizeof(glm::vec4),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &F[0],			dev_F,			numParticles*sizeof(glm::vec4),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &U[0],			dev_U,			numParticles*sizeof(glm::vec4),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &J[0],			dev_J,			numParticles*sizeof(glm::mat3),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &sigma[0],		dev_sigma,		numParticles*sizeof(glm::mat3),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &epsilon[0],	dev_epsilon,	numParticles*sizeof(glm::mat3),	cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &Acc[0],		dev_Acc,		numParticles*sizeof(glm::vec4), cudaMemcpyDeviceToHost);
}

DeformableModel *phys_model;

DeformableModel::ParallelismType type_paralellism = DeformableModel::ParallelismType::NONE;



float timeStep =  1/250.0f;
float currentTime = 0;
double accumulator = timeStep;
int selected_index = -1;

int what=0;//0-displacements, 1-stresses, 2-strains

const int width  = 1024;
const int height = 1024;
int numX = 10, numY=5, numZ=5;
const int total_points = numX*numY*numZ;

int sizeX = 4;
int sizeY = 1;
int sizeZ = 1;
float hsizeX = sizeX/2.0f;
float hsizeY = sizeY/2.0f;
float hsizeZ = sizeZ/2.0f;

int state =1 ;
float dist=-23;
const int NEIGH_SIZE = 10;
const int GRID_SIZE=10;
float pointSize = 10;
float spacing =  float(sizeX)/(numX+1);                                                 // Spacing of particles

glm::vec3 gravity=glm::vec3(0.0f,-9.81f,0.0f);

GLint viewport[4];
GLdouble MV[16];
GLdouble P[16];

bool bShowForces=false, bShowJacobians=false;

float scFac = 0; //scaling constant

int oldX=0, oldY=0;
float rX=15, rY=0;
glm::vec3 Up=glm::vec3(0,1,0), Right, viewDir;

LARGE_INTEGER frequency;        // ticks per second
LARGE_INTEGER t1, t2;           // ticks
double frameTimeQP=0;
float frameTime =0 ;

float startTime =0, fps=0;
int totalFrames=0;

int whichIndex = 0;
char info[MAX_PATH]={0};

float nu =      0.4f;			//Poisson ratio
float Y = 300000.0f;			//Young modulus
float density = 10000.f;		//material density
float kv=100, damping=500.0f;	//constant used in Eq. 22 page 5
float d15 = Y / (1.0f + nu) / (1.0f - 2 * nu);
float d16 = (1.0f - nu) * d15;
float d17 = nu * d15;
float d18 = Y / 2 / (1.0f + nu);

void OnMouseDown(int button, int s, int x, int y)
{
	if (s == GLUT_DOWN)
	{
		oldX = x;
		oldY = y;
		int window_y = (height - y);
		int window_x = x ;

		float winZ=0;
		glReadPixels( x, height-y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );
		if(winZ==1)
			winZ=0;
		double objX=0, objY=0, objZ=0;
		gluUnProject(window_x,window_y, winZ,  MV,  P, viewport, &objX, &objY, &objZ);
		glm::vec3 pt(objX,objY, objZ);
		int i=0;
		for(i=0;i<total_points;i++) {
			if( glm::distance(glm::vec3(phys_model->getPositions(type_paralellism)->at(i)),pt)<0.1) {
				selected_index = i;
				printf("Intersected at %d\n",i);
				break;
			}
		}
	}

	if(button == GLUT_MIDDLE_BUTTON)
		state = 0;
	else
		state = 1;

	if(s==GLUT_UP) {
		selected_index= -1;
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}


void OnMouseMove(int x, int y)
{
	if(selected_index == -1) {
		if (state == 0)
			dist *= (1 + (y - oldY)/60.0f);
		else
		{
			rY += (x - oldX)/5.0f;
			rX += (y - oldY)/5.0f;
		}
	} else {
		float delta = 1500/abs(dist);
		float valX = (x - oldX)/delta;
		float valY = (oldY - y)/delta;
		if(abs(valX)>abs(valY))
			glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
		else
			glutSetCursor(GLUT_CURSOR_UP_DOWN);

		vector<glm::vec4> *X = const_cast<vector<glm::vec4>*> (phys_model->getPositions(type_paralellism));
		vector<glm::vec4> *V = const_cast<vector<glm::vec4>*> (phys_model->getVelocities(type_paralellism));

		X->at(selected_index).x += Right[0]*valX;
		float newValue = X->at(selected_index).y+Up[1]*valY;
		if(newValue>0)
			X->at(selected_index).y = newValue;
		X->at(selected_index).z += Right[2]*valX + Up[2]*valY;

		V->at(selected_index).x = 0;
		V->at(selected_index).y = 0;
		V->at(selected_index).z = 0;

		if (type_paralellism == DeformableModel::CUDA) phys_model->_uploadPosVel();
	}
	oldX = x;
	oldY = y;

	glutPostRedisplay();
}


void DrawGrid()
{
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	for(int i=-GRID_SIZE;i<=GRID_SIZE;i++)
	{
		glVertex3f((float)i,0,(float)-GRID_SIZE);
		glVertex3f((float)i,0,(float)GRID_SIZE);

		glVertex3f((float)-GRID_SIZE,0,(float)i);
		glVertex3f((float)GRID_SIZE,0,(float)i);
	}
	glEnd();
}


void InitGL() {
	startTime = (float)glutGet(GLUT_ELAPSED_TIME);
	currentTime = startTime;

	// get ticks per second
	QueryPerformanceFrequency(&frequency);

	// start timer
	QueryPerformanceCounter(&t1);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
	int i=0, j=0,count=0;

	float ypos = 4.0f;

	vector<glm::vec4> X;
	X.resize(numX*numY*numZ);
	//fill in X
	for(int k=0;k<numZ;k++) {
		for( j=0;j<numY;j++) {
			for( i=0;i<numX;i++) {
				X[count++] = glm::vec4( ((float(i)/(numX-1)) )*  sizeX,
					((float(j)/(numY-1))*2-1)* hsizeY + ypos,
					((float(k)/(numZ-1))*2-1)* hsizeZ, 0.0f);

			}
		}
	}

	phys_model = new DeformableModel(X, Y, nu, false);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glPointSize(pointSize );

	wglSwapIntervalEXT(0);
}

void OnReshape(int nw, int nh) {
	glViewport(0,0,nw, nh);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)nw / (GLfloat)nh, 1.f, 100.0f);

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_PROJECTION_MATRIX, P);

	glMatrixMode(GL_MODELVIEW);
}

void OnRender() {
	int i=0;
	float newTime = (float) glutGet(GLUT_ELAPSED_TIME);
	frameTime = newTime-currentTime;
	currentTime = newTime;
	//accumulator += frameTime;

	//Using high res. counter
	QueryPerformanceCounter(&t2);
	// compute and print the elapsed time in millisecond
	frameTimeQP = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	t1=t2;
	accumulator += frameTimeQP;

	++totalFrames;
	if((newTime-startTime)>1000)
	{
		float elapsedTime = (newTime-startTime);
		fps = (totalFrames/ elapsedTime)*1000 ;
		startTime = newTime;
		totalFrames=0;
	}

	sprintf_s(info, "FPS: %3.2f, Frame time (GLUT): %3.4f ms, (QP): %3.3f ms, Young Mod.: %f, Poisson ratio: %4.4f", fps, frameTime, frameTimeQP, Y, nu);

	glutSetWindowTitle(info);
	glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0,0,dist);
	glRotatef(rX,1,0,0);
	glRotatef(rY,0,1,0);

	glGetDoublev(GL_MODELVIEW_MATRIX, MV);
	viewDir.x = (float)-MV[2];
	viewDir.y = (float)-MV[6];
	viewDir.z = (float)-MV[10];
	Right = glm::cross(viewDir, Up);

	const vector<glm::vec4> *X = phys_model->getPositions(type_paralellism);
	//draw grid
	DrawGrid();

	//draw mesh
	glColor3f(0.75,0.75,0.75);
	glBegin(GL_LINES);
	for(int i=0;i<total_points;i++) {
		for(size_t j=0;j< NEIGH_SIZE;j++) {
			glVertex3f(X->at(i).x, X->at(i).y, X->at(i).z);
			glVertex3f(X->at(phys_model->getNeighboors()->at(i*NEIGH_SIZE+j).j).x,
				X->at(phys_model->getNeighboors()->at(i*NEIGH_SIZE+j).j).y,
				X->at(phys_model->getNeighboors()->at(i*NEIGH_SIZE+j).j).z);
		}
	}
	glEnd();

	//draw points
	glBegin(GL_POINTS);

	//Debug code to see the neighbors of a given node
	glColor3f(1,1,0);

	glVertex3fv(&X->at(whichIndex).x);
	glColor3f(0,0,1);
	for(int i=0;i<NEIGH_SIZE;i++) {
		int neigh(phys_model->getNeighboors()->at(whichIndex*NEIGH_SIZE+i).j);
		if(neigh !=whichIndex) {
			glVertex3fv(&X->at(neigh).x);
		}
	}
	for(i=0;i<total_points;i++) {
		glm::vec4 p = X->at(i);
		if(i==selected_index){
			glColor3f(0,1,1);
			glVertex3f(p.x,p.y,p.z);
		}
		else{
			glColor3f(1,0,0);
			glVertex3f(p.x,p.y,p.z);
		}
	}
	glEnd();

	glutSwapBuffers();
}

void OnShutdown() {
	delete phys_model;
}


void OnIdle() {
	//Fixed time stepping + rendering at different fps
	if ( accumulator >= timeStep )
	{
		phys_model->stepPhysics(timeStep, type_paralellism);
		accumulator -= timeStep;
	}

	glutPostRedisplay();
}

void OnKey(unsigned char k,int , int) {
	switch(k) {
	case 'a':whichIndex--;break;
	case 'd':whichIndex++;break;
	case 'f':bShowForces=!bShowForces;break;
	case 'j':bShowJacobians=!bShowJacobians;break;
	case 'e':what=2;break;
	case 's':what=1;break;
	case 'x':what=0;break;
	case ',':Y-=500;break;
	case '.':Y+=500;break;
	case '[':nu-=0.01f;break;
	case ']':nu+=0.01f;break;
	case '0':type_paralellism = DeformableModel::ParallelismType::NONE;break;
	case '1':type_paralellism = DeformableModel::ParallelismType::OPENMP;break;
	case '2':type_paralellism = DeformableModel::ParallelismType::CUDA;break;
	case ' ':phys_model->togglePaused();break;
	}

	whichIndex = (whichIndex%total_points);
	glutPostRedisplay();

}
int main(int argc, char** argv) {
	atexit(OnShutdown);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Luis Yanes - Demo GPU application");

	glutDisplayFunc(OnRender);
	glutReshapeFunc(OnReshape);
	glutIdleFunc(OnIdle);

	glutMouseFunc(OnMouseDown);
	glutMotionFunc(OnMouseMove);

	glutKeyboardFunc(OnKey);
	glewInit();
	InitGL();

	puts("Press 'u' or 'U' to decrease/increase Poisson Ratio (nu)");
	puts("Press 'y' or 'Y' to decrease/increase Young's Modulus (Y)");
	puts("Press 'j' to display the Jacobian");
	puts("Press 'f' to view the Forces\n");
	puts("\tPress 'x' for displacement, 'e' for strains and 's' for stresses");

	glutMainLoop();

	return EXIT_SUCCESS;
}