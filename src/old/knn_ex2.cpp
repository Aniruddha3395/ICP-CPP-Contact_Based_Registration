// This example is in the public domain
#include <iostream>
#include <Eigen/Eigen>
#include "nabo/nabo.h"
#include <chrono>

	Nabo::NNSearchD * dummy(Eigen::MatrixXd Mt){
		// std::cout << "In Main Func!!" << std::endl;
		
		// Eigen::MatrixXd M(Mt.cols(),Mt.rows());
		// M << Mt.transpose();
		// std::cout << M.rows() << std::endl;
		// std::cout << M.cols() << std::endl;
		
		// create a kd-tree for M, note that M must stay valid during the lifetime of the kd-tree
		Nabo::NNSearchD* nns = Nabo::NNSearchD::createKDTreeLinearHeap(Mt);
		int dim = nns->dim;
		// std::cout << nns << std::endl;
		return nns;
	}



int main()
{

	Eigen::MatrixXd Mt = Eigen::MatrixXd::Random(3,1000000);

	Nabo::NNSearchD * nns_temp;
	nns_temp = dummy(Mt);
		int dim = nns_temp->dim;
		// std::cout << nns_temp << std::endl;

	// std::cout << "tree made" << std::endl;

	// for (int i=0;i<10000;++i)
	// {
	Eigen::MatrixXd q = Eigen::MatrixXd::Random(3,20);

	const int K = 1;
	Eigen::MatrixXi indices(K,q.cols());
	Eigen::MatrixXd dists2(K,q.cols());

	auto gpu_start = std::chrono::high_resolution_clock::now();
	dummy(Mt)->knn(q, indices, dists2, K);
	auto gpu_end = std::chrono::high_resolution_clock::now();
   	std::cout << "vector_add_gpu time: " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() << " milliseconds.\n";

	std::cout << indices << std::endl;
	// }

	// // cleanup kd-tree
	// delete nns_temp;

	// std::cout << "Ending Main Func!!" << std::endl;
	
	return 0;
}



	