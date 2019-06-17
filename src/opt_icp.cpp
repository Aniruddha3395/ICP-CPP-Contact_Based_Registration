#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "nabo/nabo.h"
#include "nlopt.hpp"
#include "stdlib.h"
#include "utilities.hpp"
#include "file_rw.hpp"
#include "transformation_utilities.hpp"
#include "opt_icp.hpp"
#include <chrono>

namespace opt_icp
{
    // defualt nlopt fuction
    double customminfunc(const std::vector<double>& x, std::vector<double>& grad, void* data) {
        // Auxilory function to minimize (Sum of Squared distance between correspondance
        // from both pointclouds). Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.
        opt_icp *c = (opt_icp *) data;
        return c->ObjFun(x,grad);
    };

    double customconfunc(const std::vector<double>& x, std::vector<double>& grad, void* data) 
    {
        // Because we wanted a Class
        // without static members, but NLOpt library does not support
        // passing methods of Classes, we use these auxilary functions.
        opt_icp *d = (opt_icp *) data;
        return d->ConFun(x,grad);
    };

    // class constructor
    opt_icp::opt_icp(std::vector<double> x_start, int K, Eigen::MatrixXd Input_Part_Ptcloud_Icp,
        Eigen::MatrixXd Scanned_Traj, const double error_threshold, double Perturb_Val,
        double w1, double w2, double OptH, double optXtolRel) 
    {
        //choose optimizer
        // optalg = nlopt::LN_NEWUOA;
        // optalg = nlopt::LN_NEWUOA_BOUND;
        // optalg = nlopt::LN_BOBYQA;
        // optalg = nlopt::LN_COBYLA;
        // optalg = nlopt::GN_ISRES;
        optalg = nlopt::LD_SLSQP;
        
        k = K;
        W1 = w1; 
        W2 = w2;
        optH = OptH;
        OptXtolRel = optXtolRel;
        input_part_ptcloud_icp = Input_Part_Ptcloud_Icp;
        scanned_traj = Scanned_Traj;
        x0 = x_start;
        Error_threshold = error_threshold;
        perturb_val = Perturb_Val;
        corresponding_val_from_part_ptcloud = Eigen::MatrixXd(k,3);
        x_vec = Eigen::MatrixXd(k,1);
        y_vec = Eigen::MatrixXd(k,1);
        z_vec = Eigen::MatrixXd(k,1);

        M = Eigen::MatrixXd(input_part_ptcloud_icp.cols(),input_part_ptcloud_icp.rows());
        M << input_part_ptcloud_icp.transpose();
        
        // create a kd-tree for M, note that M must stay valid during the lifetime of the kd-tree
        Nabo::NNSearchD * nns_temp = Nabo::NNSearchD::createKDTreeLinearHeap(M);
        nns = nns_temp;

        // optimization params
        OptVarDim = x_start.size();
        opt = nlopt::opt(optalg, OptVarDim);
        OptVarlb.resize(OptVarDim);
        OptVarub.resize(OptVarDim);

        OptVarlb[0] = -20; OptVarub[0] = 20;
        OptVarlb[1] = -20; OptVarub[1] = 20;
        OptVarlb[2] = -20; OptVarub[2] = 20;
        OptVarlb[3] = -1;  OptVarub[3] = 1;
        OptVarlb[4] = -1;  OptVarub[4] = 1;
        OptVarlb[5] = -1;  OptVarub[5] = 1;
        OptVarlb[6] = -1;  OptVarub[6] = 1;
        
        opt.set_xtol_rel(OptXtolRel);
        opt.set_min_objective(customminfunc, this);
        opt.add_equality_constraint(customconfunc, this, 1e-8);
        
        opt.set_upper_bounds(OptVarub);
        opt.set_lower_bounds(OptVarlb);
        opt.set_ftol_rel(1e-8);
        opt.set_maxeval(1000);
        
        optx.resize(OptVarDim);

        srand(time(0));
    };

    // class destructor
    opt_icp::~opt_icp()
    {
        delete nns;
    };

    // non-linear equality constraints
    double opt_icp::NLConFun(const std::vector<double> &x)
    {
        return (x[3]*x[3] + x[4]*x[4] + x[5]*x[5] + x[6]*x[6] -1);
    };
    
    // main error function for computing the weighted max-avg distance between ptclouds    
    double opt_icp::ErrFun(const std::vector<double> &x)
    {        
        double E = 0;
        Eigen::MatrixXd r(1,4);
        r << x[3],x[4],x[5],x[6];
        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
        transformation_matrix(0,3) = x[0];transformation_matrix(1,3) = x[1];transformation_matrix(2,3) = x[2];
        transformation_matrix.block(0,0,3,3) = rtf::qt2rot(r);
        Eigen::MatrixXd transformation_matrix_new = transformation_matrix.inverse();
        Eigen::MatrixXd transformed_data = rtf::apply_transformation(scanned_traj,transformation_matrix_new);

        double d[scanned_traj.rows()];
        long double sum_d = 0.0;
        double max_d = -10000000;
        
        Eigen::MatrixXd q = scanned_traj.transpose();
        Eigen::MatrixXi indices(k, q.cols());
        Eigen::MatrixXd dists(k, q.cols());
        
        nns->knn(q, indices, dists, k);

        double x_avg, y_avg, z_avg, L00, L11, L01, R0, R1, A, B, C, D; 
            
        for (long int i=0;i<scanned_traj.rows();++i)
        {
            for (int j=0;j<k;++j)
            {
                corresponding_val_from_part_ptcloud.row(j) = input_part_ptcloud_icp.row(indices(j,i));
            }

            // least square plane fitting and distance between point and plane
            querry_pt = transformed_data.row(i);
            x_vec = corresponding_val_from_part_ptcloud.col(0);
            y_vec = corresponding_val_from_part_ptcloud.col(1);
            z_vec = corresponding_val_from_part_ptcloud.col(2);
            x_avg = x_vec.sum()/corresponding_val_from_part_ptcloud.rows();
            y_avg = y_vec.sum()/corresponding_val_from_part_ptcloud.rows();
            z_avg = z_vec.sum()/corresponding_val_from_part_ptcloud.rows();
            L00 = ((x_vec.array() - x_avg).array().pow(2)).sum();
            L01 = ((x_vec.array() - x_avg).array()*(y_vec.array() - y_avg).array()).sum(); 
            L11 = ((y_vec.array() - y_avg).array().pow(2)).sum();
            R0 = ((z_vec.array() - z_avg).array()*(x_vec.array() - x_avg).array()).sum(); 
            R1 = ((z_vec.array() - z_avg).array()*(y_vec.array() - y_avg).array()).sum(); 
            A = -((L11*R0-L01*R1)/(L00*L11-L01*L01));
            B = -((L00*R1-L01*R0)/(L00*L11-L01*L01));
            C = 1;
            D = -(z_avg+A*x_avg+B*y_avg);
            d[i] = std::abs(A*querry_pt(0,0)+B*querry_pt(0,1)+C*querry_pt(0,2)+D)/(sqrt(A*A+B*B+C*C));
            sum_d += d[i];
            if (d[i] > max_d)
            {
                max_d=d[i];
            }
        }

        // error calculation
        E = ((W1*sum_d)/scanned_traj.rows()) + (W2*max_d);
        return E;
    };

    Eigen::MatrixXd opt_icp::solveOPT()
    {
        solx = x0;
        int successFlag = 0;
        try{
            nlopt::result result = opt.optimize(solx, solminf);
            successFlag = 1;
        }
        catch(std::exception &e) {
            std::cout << "nlopt failed: " << e.what() << std::endl;
        }
        Eigen::MatrixXd opt_result(1,9);
        for (int i=0;i<solx.size();++i)
        {
            opt_result(0,i) = solx[i];
        }
        opt_result(0,7) = solminf;
        opt_result(0,8) = successFlag;

        return opt_result;
    };

    

    // gradient computation for error function
    // Forward Difference Method
    double opt_icp::ObjFun(const std::vector<double> &x, std::vector<double> &grad)
    {
        double err = ErrFun(x);
        if (!grad.empty()) {
            std::vector<double> xph = x;
            for (uint i=0; i < x.size(); ++i)
            {
                xph[i] += optH;
                grad[i] = (ErrFun(xph)-err)/optH;
                xph[i] -= optH;
            }
        }    
        return err;
    };

    // gradient computation for constraints
    // Forward Difference Method
    double opt_icp::ConFun(const std::vector<double> &x, std::vector<double> &grad)
    {
        double err = NLConFun(x);
        if (!grad.empty()) {
            std::vector<double> xph = x;
            for (uint i=0; i < x.size(); ++i)
            {
                xph[i] += optH;
                grad[i] = (NLConFun(xph)-err)/optH;
                xph[i] -= optH;
            }
        }    
        return err;
    };
}


//==================================================================================================================//
//==================================================================================================================//
//==========================================Backup Error Function===================================================//
//==================================================================================================================//
//==================================================================================================================//

// double opt_icp::ErrFun(const std::vector<double> &x)
//     {
        
//         double E = 0;
//         Eigen::MatrixXd r(1,4);
//         r << x[3],x[4],x[5],x[6];
//         Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
//         transformation_matrix(0,3) = x[0];transformation_matrix(1,3) = x[1];transformation_matrix(2,3) = x[2];
//         transformation_matrix.block(0,0,3,3) = rtf::qt2rot(r);
        
//         auto t_s1 = std::chrono::high_resolution_clock::now();
//         Eigen::MatrixXd transformed_data = rtf::apply_transformation(input_part_ptcloud_icp,transformation_matrix);
//         auto t_e1 = std::chrono::high_resolution_clock::now();

//         // Eigen::MatrixXd q(3,1);
//         // Eigen::VectorXi indices(k);
//         // Eigen::VectorXd dists(k);

//         double d[scanned_traj.rows()];
//         long double sum_d = 0.0;
//         double max_d = -10000000;
        
//         Eigen::MatrixXd q = scanned_traj.transpose();
//         Eigen::MatrixXi indices(k, q.cols());
//         Eigen::MatrixXd dists(k, q.cols());
        
//         auto t_s2 = std::chrono::high_resolution_clock::now();
//         nns->knn(q, indices, dists, k);
//         auto t_e2 = std::chrono::high_resolution_clock::now();

//         auto t_s3 = std::chrono::high_resolution_clock::now();
        
//         double x_avg, y_avg, z_avg, L00, L11, L01, R0, R1, A, B, C, D; 
            
//         for (long int i=0;i<scanned_traj.rows();++i)
//         {
//             for (int j=0;j<k;++j)
//             {
//                 corresponding_val_from_part_ptcloud.row(j) = transformed_data.row(indices(j,i));
//             }

//             // d[i] = ut::get_pt_to_lsf_plane_dist(q.col(i).transpose(),corresponding_val_from_part_ptcloud);
//             querry_pt = q.col(i).transpose();
//             x_vec = corresponding_val_from_part_ptcloud.col(0);
//             y_vec = corresponding_val_from_part_ptcloud.col(1);
//             z_vec = corresponding_val_from_part_ptcloud.col(2);
//             x_avg = x_vec.sum()/corresponding_val_from_part_ptcloud.rows();
//             y_avg = y_vec.sum()/corresponding_val_from_part_ptcloud.rows();
//             z_avg = z_vec.sum()/corresponding_val_from_part_ptcloud.rows();
//             L00 = ((x_vec.array() - x_avg).array().pow(2)).sum();
//             L01 = ((x_vec.array() - x_avg).array()*(y_vec.array() - y_avg).array()).sum(); 
//             L11 = ((y_vec.array() - y_avg).array().pow(2)).sum();
//             R0 = ((z_vec.array() - z_avg).array()*(x_vec.array() - x_avg).array()).sum(); 
//             R1 = ((z_vec.array() - z_avg).array()*(y_vec.array() - y_avg).array()).sum(); 
//             A = -((L11*R0-L01*R1)/(L00*L11-L01*L01));
//             B = -((L00*R1-L01*R0)/(L00*L11-L01*L01));
//             C = 1;
//             D = -(z_avg+A*x_avg+B*y_avg);
//             d[i] = std::abs(A*querry_pt(0,0)+B*querry_pt(0,1)+C*querry_pt(0,2)+D)/(sqrt(A*A+B*B+C*C));

//             sum_d += d[i];
//             if (d[i] > max_d)
//             {
//                 max_d=d[i];
//             }
//         }

//         E = ((W1*sum_d)/scanned_traj.rows()) + (W2*max_d);
//         auto t_e3 = std::chrono::high_resolution_clock::now();
               
//         // std::cout << E << std::endl;
//         std::cout << E << " time 1 : " << std::chrono::duration_cast<std::chrono::microseconds>(t_e1 - t_s1).count() << " mis"
//             << " time 2 : " << std::chrono::duration_cast<std::chrono::microseconds>(t_e2 - t_s2).count() << " mis"
//             << " time 3 : " << std::chrono::duration_cast<std::chrono::microseconds>(t_e3 - t_s3).count() << " mis\n";        
//         return E;
//     };