#include "nabo/nabo.h"
#include "nlopt.hpp"
#include <Eigen/Eigen>

namespace opt_icp
{

    class opt_icp
    {
        private:

        public:
            // constructor and distructor
            opt_icp(std::vector<double> x_start, int K, Eigen::MatrixXd Input_Part_Ptcloud_Icp, 
                Eigen::MatrixXd Scanned_Traj, const double error_threshold, double Perturb_Val, 
                double w1, double w2, double OptH, double optXtolRel);
            ~opt_icp();

            // other variables
            double W1;
            double W2;
            Nabo::NNSearchD * nns;
            int k;
            double Error_threshold;
            double perturb_val;
            Eigen::MatrixXd input_part_ptcloud_icp;
            Eigen::MatrixXd scanned_traj;
            Eigen::MatrixXd M;
            std::vector<double> x0;

            // optim solver
            int OptVarDim;
            double OptXtolRel;
            double optH;
            std::vector<double> optx;
            std::vector<double> OptVarlb;
            std::vector<double> OptVarub;        
            std::vector<double> solx;
            double solminf;
            nlopt::algorithm optalg;
            nlopt::opt opt;

            Eigen::MatrixXd corresponding_val_from_part_ptcloud;
        
            Eigen::MatrixXd x_vec;
            Eigen::MatrixXd y_vec;
            Eigen::MatrixXd z_vec;
            Eigen::MatrixXd querry_pt{1,3};
            
            double ErrFun(const std::vector<double> &x);
            double NLConFun(const std::vector<double> &x);
            double ObjFun(const std::vector<double> &x, std::vector<double> &grad);
            double ConFun(const std::vector<double> &x, std::vector<double> &grad);
            Eigen::MatrixXd solveOPT();
    };
}