#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <ctime>
#include "nabo/nabo.h"
#include "nlopt.h"
#include "stdlib.h"
#include "utilities.hpp"
#include "file_rw.hpp"
#include "transformation_utilities.hpp"
#include "opt_icp.hpp"

int main(int argc, char const *argv[])
{

bool include_input_ptclouds_testing = false;

////////////////////////////GLOBAL CONFIG////////////////////////////////////
    std::string Data_dir = "/home/aniruddha/Desktop/ICP/ICP_OPT/data/";    
    std::string data_num;
    if (argc==2)
    {
        data_num = argv[1];
    }
    else
    {
        data_num = "1";    
    }

    // Tool Data
    Eigen::Vector3d tool_t;
    Eigen::MatrixXd tool_r(1,3);
    Eigen::Matrix4d tool_F_T_tcp;
    if (data_num.compare("1")==0)
    {
        tool_t << -0.2,-0.01,48.6; //values in mm
    }
    else
    {
        tool_t << 1.7,0.7,45.6; //values in mm
    }

    tool_r << 0,0,0;    // values in Euler ZYX
    
    tool_F_T_tcp = rtf::hom_T(tool_t,rtf::eul2rot(tool_r,"ZYX"));

    // Input Transformation Matrix for ICP
    // Eigen::Vector3d input_w_t_p;
    // Eigen::MatrixXd input_w_r_p(1,3);
    // Eigen::Matrix4d input_w_T_p;
    // input_w_t_p << 482.4196,-148.6004,306.3996;       // values in mm
    // input_w_r_p << 0.0350,0.0352,-0.0887;        // values in Euler ZYX

    // input_w_T_p = rtf::hom_T(input_w_t_p,rtf::eul2rot(input_w_r_p,"ZYX"));
    std::string input_T_file = Data_dir+"data"+data_num+"/transform_mat.csv";
    Eigen::MatrixXd input_w_T_p = file_rw::file_read_mat(input_T_file);    
    // std::cout << input_w_T_p << std::endl;

    // Part pointcloud from STL file 
    std::string part_ptcloud_file = Data_dir+"data"+data_num+"/part_ptcloud.csv";
    std::vector<std::vector<double> > part_ptcloud_vec;
    part_ptcloud_vec = file_rw::file_read_vec(part_ptcloud_file);
    Eigen::MatrixXd part_ptcloud(part_ptcloud_vec.size(),part_ptcloud_vec[0].size());
    part_ptcloud = ut::vec_to_mat(part_ptcloud_vec);

    // Part poinrcoud transformed using Input Transformation Matrix for ICP
    Eigen::MatrixXd input_part_ptcloud_icp(part_ptcloud.rows(),part_ptcloud.cols());
    input_part_ptcloud_icp = rtf::apply_transformation(part_ptcloud, input_w_T_p);

    
    // Data for ICP from KUKA Scanning
    std::string traj_from_kuka_scanning_file = Data_dir+"data"+data_num+"/data_for_ICP.csv";
    std::vector<std::vector<double> > traj_from_kuka_scanning_vec;
    traj_from_kuka_scanning_vec = file_rw::file_read_vec(traj_from_kuka_scanning_file);
    Eigen::MatrixXd scan_traj_wrt_tcp(traj_from_kuka_scanning_vec.size(),traj_from_kuka_scanning_vec[0].size());
    scan_traj_wrt_tcp = ut::get_traj_wrt_tcp(tool_F_T_tcp, traj_from_kuka_scanning_vec);
 
    // std::vector<std::vector<double> > part_ptcloud_normals_vec;
    // std::string part_ptcloud_normals_file = "/home/rflin/Desktop/ANIRUDDHA_WS/CPP/data_files/part_ptcloud_normals.csv";
    // std::string scanned_traj_file = "/home/rflin/Desktop/ANIRUDDHA_WS/CPP/data_files/scanned_traj.csv";
    // std::vector<std::vector<double> > scanned_traj_vec;
    // part_ptcloud_normals_vec = file_rw::file_read(part_ptcloud_normals_file);
    // scanned_traj_vec = file_rw::file_read(scanned_traj_file);
    // Eigen::MatrixXd part_ptcloud_normals(part_ptcloud_normals_vec.size(),part_ptcloud_normals_vec[0].size());
    // Eigen::MatrixXd scanned_traj(scanned_traj_vec.size(),scanned_traj_vec[0].size());
    // part_ptcloud_normals = ut::vec_to_mat(part_ptcloud_normals_vec);
    // scanned_traj = ut::vec_to_mat(scanned_traj_vec);
    // Eigen::MatrixXd input_part_ptcloud_icp_true(input_part_ptcloud_icp.rows(),input_part_ptcloud_icp.cols());
    // input_part_ptcloud_icp_true = input_part_ptcloud_icp;

    // Optimization Params 
    const int K = 5;            // K is number of nearest neighbours to be calculated for the lsf plane
    double Error_threshold = 0.1;  // min error value for distance in meters
    double perturb_val = 0.6;     //value in meters
    std::vector<double> x0 = {0,0,0,0,0,0,1};       //Seed for ICP
    double W1 = 0.75;       // Weight for Avg(dist)
    double W2 = 0.25;       // Weight for Max(dist)
    double optH = 1e-10;
    double OptXtolRel = 1e-10;

    // Optimization Routine
    // Eigen::Matrix4d icp_T;
    // Eigen::Vector3d icp_t;
    // Eigen::MatrixXd icp_qt(1,4);
    // Eigen::MatrixXd icp_eul(1,3);
    // Eigen::Matrix4d icp_T_final = Eigen::Matrix4d::Identity();
    // Eigen::Matrix4d icp_T_final_save;
    // Eigen::Matrix4d Final_w_T_p;


    if (include_input_ptclouds_testing)
    {
        std::string ptcloud1_file = Data_dir+"data"+data_num+"/ptcloud1.csv";
        Eigen::MatrixXd ptcloud1 = file_rw::file_read_mat(ptcloud1_file);
        std::string ptcloud2_file = Data_dir+"data"+data_num+"/ptcloud2.csv";
        Eigen::MatrixXd ptcloud2 = file_rw::file_read_mat(ptcloud2_file);
        std::cout << input_part_ptcloud_icp.array()-ptcloud1.array() << std::endl;
        std::cout << scan_traj_wrt_tcp.array()-ptcloud2.array() << std::endl;
        return 0;
    }
    // std::cout << input_part_ptcloud_icp.block(0,0,20,input_part_ptcloud_icp.cols()) << std::endl << std::endl;
    // std::cout << scan_traj_wrt_tcp.block(0,0,20,scan_traj_wrt_tcp.cols()) << std::endl << std::endl;

    // Eigen::MatrixXd input_part_ptcloud_icp_save(input_part_ptcloud_icp.rows(),input_part_ptcloud_icp.cols());
    // opt_icp::opt_icp OptObj(x0, K, input_part_ptcloud_icp, scan_traj_wrt_tcp, 
    //     Error_threshold, perturb_val, W1, W2, optH, OptXtolRel);

    std::cout << "initialization done quickly" << std::endl;
    // Eigen::MatrixXd opt_result = OptObj.solveOPT();
    // std::cout << opt_result <<std::endl;
    // return 0;
    // icp_T_final_save = OptObj.solveOPTCust();        


        Eigen::Matrix4d icp_T;
        Eigen::Vector3d icp_t;
        Eigen::MatrixXd icp_qt(1,4);
        Eigen::MatrixXd icp_eul(1,3);
        Eigen::Matrix4d icp_T_final = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d icp_T_final_save;
        Eigen::Matrix4d Final_w_T_p;
        Eigen::MatrixXd input_part_ptcloud_icp_save(input_part_ptcloud_icp.rows(),input_part_ptcloud_icp.cols());
        Eigen::MatrixXd opt_result(1,9);

        double fval_curr = 1e6;
        int counter = 0;
        while(fval_curr>Error_threshold && counter<15)
        {
            // optimization routine
            opt_icp::opt_icp OptObj(x0, K, input_part_ptcloud_icp, scan_traj_wrt_tcp, 
                Error_threshold, perturb_val, W1, W2, optH, OptXtolRel);
            opt_result = OptObj.solveOPT();

            icp_t << opt_result(0,0),opt_result(0,1),opt_result(0,2);
            icp_qt << opt_result(0,3),opt_result(0,4),opt_result(0,5),opt_result(0,6);
            icp_T = rtf::hom_T(icp_t,rtf::qt2rot(icp_qt));
            if (opt_result(0,7)<fval_curr)
            {
                fval_curr=opt_result(0,7);
                for (int j=0;j<x0.size();++j)
                {
                    x0[j] = opt_result(0,j);    
                }
                icp_T_final = icp_T*icp_T_final;
                input_part_ptcloud_icp = rtf::apply_transformation(input_part_ptcloud_icp,icp_T);
                icp_T_final_save = icp_T_final;
                input_part_ptcloud_icp_save = input_part_ptcloud_icp;
                // std::cout << fval_curr << "," << solminf << std::endl;
            }
            else
            {
            	x0 = {0,0,0,0,0,0,1};
                icp_t(0) = opt_result(0,0) - perturb_val + 2*perturb_val*((double)rand() /(double)(RAND_MAX));
                icp_t(1) = opt_result(0,1) - perturb_val + 2*perturb_val*((double)rand() /(double)(RAND_MAX));
                icp_t(2) = opt_result(0,2) - perturb_val + 2*perturb_val*((double)rand() /(double)(RAND_MAX));
                icp_eul = rtf::qt2eul(icp_qt);
                // icp_eul(0,0) = (icp_eul(0,0) - 0.5*perturb_val + 1*perturb_val*((double) rand() / (double)(RAND_MAX)))*(3.14159/180);
                // icp_eul(0,1) = (icp_eul(0,1) - 0.5*perturb_val + 1*perturb_val*((double) rand() / (double)(RAND_MAX)))*(3.14159/180);
                // icp_eul(0,2) = (icp_eul(0,2) - 0.5*perturb_val + 1*perturb_val*((double) rand() / (double)(RAND_MAX)))*(3.14159/180);
                
                icp_eul(0,0) = icp_eul(0,0) + (- 0.5*perturb_val + 1*perturb_val*((double)rand() /(double)(RAND_MAX)))*(3.14159/180);
                icp_eul(0,1) = icp_eul(0,1) + (- 0.5*perturb_val + 1*perturb_val*((double)rand() /(double)(RAND_MAX)))*(3.14159/180);
                icp_eul(0,2) = icp_eul(0,2) + (- 0.5*perturb_val + 1*perturb_val*((double)rand() /(double)(RAND_MAX)))*(3.14159/180);
                
                icp_T = rtf::hom_T(icp_t,rtf::eul2rot(icp_eul));
                icp_T_final = icp_T*icp_T_final;
                input_part_ptcloud_icp = rtf::apply_transformation(input_part_ptcloud_icp, icp_T);
            }
            std::cout << fval_curr << "," << opt_result(0,7) << std::endl;
            counter++;
        }    
        // return icp_T_final_save;





    // Final Transformation Matrix
    Final_w_T_p = icp_T_final_save*input_w_T_p;
    std::cout << Final_w_T_p << std::endl;
    return 0;
}




// int main(int argc, char const *argv[])
// {

// bool include_input_ptclouds_testing = false;

// ////////////////////////////GLOBAL CONFIG////////////////////////////////////
//     std::string Data_dir = "/home/aniruddha/Desktop/ICP/ICP_OPT_testing/data/";    
//     std::string data_num;
//     if (argc==2)
//     {
//         data_num = argv[1];
//     }
//     else
//     {
//         data_num = "1";    
//     }

//     // Tool Data
//     Eigen::Vector3d tool_t;
//     Eigen::MatrixXd tool_r(1,3);
//     Eigen::Matrix4d tool_F_T_tcp;
//     if (data_num.compare("1")==0)
//     {
//         tool_t << -0.2,-0.01,48.6; //values in mm
//     }
//     else
//     {
//         tool_t << 1.7,0.7,45.6; //values in mm
//     }

//     tool_r << 0,0,0;    // values in Euler ZYX
    
//     tool_F_T_tcp = rtf::hom_T(tool_t,rtf::eul2rot(tool_r,"ZYX"));

//     // Input Transformation Matrix for ICP
//     // Eigen::Vector3d input_w_t_p;
//     // Eigen::MatrixXd input_w_r_p(1,3);
//     // Eigen::Matrix4d input_w_T_p;
//     // input_w_t_p << 482.4196,-148.6004,306.3996;       // values in mm
//     // input_w_r_p << 0.0350,0.0352,-0.0887;        // values in Euler ZYX

//     // input_w_T_p = rtf::hom_T(input_w_t_p,rtf::eul2rot(input_w_r_p,"ZYX"));
//     std::string input_T_file = Data_dir+"data"+data_num+"/transform_mat.csv";
//     Eigen::MatrixXd input_w_T_p = file_rw::file_read_mat(input_T_file);    
//     // std::cout << input_w_T_p << std::endl;

//     // Part pointcloud from STL file 
//     std::string part_ptcloud_file = Data_dir+"data"+data_num+"/part_ptcloud.csv";
//     std::vector<std::vector<double> > part_ptcloud_vec;
//     part_ptcloud_vec = file_rw::file_read_vec(part_ptcloud_file);
//     Eigen::MatrixXd part_ptcloud(part_ptcloud_vec.size(),part_ptcloud_vec[0].size());
//     part_ptcloud = ut::vec_to_mat(part_ptcloud_vec);

//     // Part poinrcoud transformed using Input Transformation Matrix for ICP
//     Eigen::MatrixXd input_part_ptcloud_icp(part_ptcloud.rows(),part_ptcloud.cols());
//     input_part_ptcloud_icp = rtf::apply_transformation(part_ptcloud, input_w_T_p);

    
//     // Data for ICP from KUKA Scanning
//     std::string traj_from_kuka_scanning_file = Data_dir+"data"+data_num+"/data_for_ICP.csv";
//     std::vector<std::vector<double> > traj_from_kuka_scanning_vec;
//     traj_from_kuka_scanning_vec = file_rw::file_read_vec(traj_from_kuka_scanning_file);
//     Eigen::MatrixXd scan_traj_wrt_tcp(traj_from_kuka_scanning_vec.size(),traj_from_kuka_scanning_vec[0].size());
//     scan_traj_wrt_tcp = ut::get_traj_wrt_tcp(tool_F_T_tcp, traj_from_kuka_scanning_vec);
 
//     // std::vector<std::vector<double> > part_ptcloud_normals_vec;
//     // std::string part_ptcloud_normals_file = "/home/rflin/Desktop/ANIRUDDHA_WS/CPP/data_files/part_ptcloud_normals.csv";
//     // std::string scanned_traj_file = "/home/rflin/Desktop/ANIRUDDHA_WS/CPP/data_files/scanned_traj.csv";
//     // std::vector<std::vector<double> > scanned_traj_vec;
//     // part_ptcloud_normals_vec = file_rw::file_read(part_ptcloud_normals_file);
//     // scanned_traj_vec = file_rw::file_read(scanned_traj_file);
//     // Eigen::MatrixXd part_ptcloud_normals(part_ptcloud_normals_vec.size(),part_ptcloud_normals_vec[0].size());
//     // Eigen::MatrixXd scanned_traj(scanned_traj_vec.size(),scanned_traj_vec[0].size());
//     // part_ptcloud_normals = ut::vec_to_mat(part_ptcloud_normals_vec);
//     // scanned_traj = ut::vec_to_mat(scanned_traj_vec);
//     // Eigen::MatrixXd input_part_ptcloud_icp_true(input_part_ptcloud_icp.rows(),input_part_ptcloud_icp.cols());
//     // input_part_ptcloud_icp_true = input_part_ptcloud_icp;

//     // Optimization Params 
//     const int K = 5;            // K is number of nearest neighbours to be calculated for the lsf plane
//     double Error_threshold = 1;  // min error value for distance in meters
//     double perturb_val = 0.5;     //value in meters
//     std::vector<double> x0 = {0,0,0,0,0,0,1};       //Seed for ICP
//     double W1 = 0.75;       // Weight for Avg(dist)
//     double W2 = 0.25;       // Weight for Max(dist)
//     double optH = 1e-10;
//     double OptXtolRel = 1e-10;

//     // Optimization Routine
//     Eigen::Matrix4d icp_T;
//     Eigen::Vector3d icp_t;
//     Eigen::MatrixXd icp_qt(1,4);
//     Eigen::MatrixXd icp_eul(1,3);
//     Eigen::Matrix4d icp_T_final = Eigen::Matrix4d::Identity();
//     Eigen::Matrix4d icp_T_final_save;
//     Eigen::Matrix4d Final_w_T_p;


//     if (include_input_ptclouds_testing)
//     {
//         std::string ptcloud1_file = Data_dir+"data"+data_num+"/ptcloud1.csv";
//         Eigen::MatrixXd ptcloud1 = file_rw::file_read_mat(ptcloud1_file);
//         std::string ptcloud2_file = Data_dir+"data"+data_num+"/ptcloud2.csv";
//         Eigen::MatrixXd ptcloud2 = file_rw::file_read_mat(ptcloud2_file);
//         std::cout << input_part_ptcloud_icp.array()-ptcloud1.array() << std::endl;
//         std::cout << scan_traj_wrt_tcp.array()-ptcloud2.array() << std::endl;
//         return 0;
//     }
//     // std::cout << input_part_ptcloud_icp.block(0,0,20,input_part_ptcloud_icp.cols()) << std::endl << std::endl;
//     // std::cout << scan_traj_wrt_tcp.block(0,0,20,scan_traj_wrt_tcp.cols()) << std::endl << std::endl;

//     Eigen::MatrixXd input_part_ptcloud_icp_save(input_part_ptcloud_icp.rows(),input_part_ptcloud_icp.cols());
//     opt_icp::opt_icp OptObj(x0, K, input_part_ptcloud_icp, scan_traj_wrt_tcp, 
//         Error_threshold, perturb_val, W1, W2, optH, OptXtolRel);
//     icp_T_final_save = OptObj.solveOPTCust();        

//     // Final Transformation Matrix
//     Final_w_T_p = icp_T_final_save*input_w_T_p;
//     std::cout << Final_w_T_p << std::endl;
//     return 0;
// }