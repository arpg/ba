/*
 *  Init
 *  SetGravity
 *  AddPose
 *	AddUnaryConstraint
 *	AddBinaryConstraint
 *	AddImuResidual
 *	  GetRange returns vector
 *	  InterpolationBuffer
 *	Solve (no dogleg)
 *
 * 	GetPose()
 *
 **/

//=============================================================================

#include <vector>
#include <thread>
#include <cfloat>

#include <BA/BundleAdjuster.h>
#include <BA/Types.h>
#include <Eigen/Eigen>

using std::vector;
using namespace std;

//=============================================================================
ba::BundleAdjuster<double,0,6,0>  slam;

vector<unsigned int> nodes;

vector<Eigen::Vector3d> gps;

/* 2D coordinate transformations */
double incremental_x = 0;
double incremental_y = 0;
double incremental_yaw = 0;
double incremental_timestamp = 0;

Sophus::SE3d incremental_tfm;
Sophus::SE3d incremental_pose;

FILE* differential;

//=============================================================================
//=============================================================================
void update_incremental_pose(double timestamp, double rr, double rl)
{
	static bool firsttime = true;
	if (firsttime)
	{
		incremental_timestamp = timestamp;
		firsttime = false;
		return;
	}

	double dt = timestamp - incremental_timestamp;
	
	const double trackwidth = 1.5;

	double TINY = 0.0001;
	if (fabs(rr) > TINY or fabs(rl) > TINY)
	{
		if (fabs(rr-rl) < TINY)
		{
			//incremental_x -= sin(incremental_yaw) * rr * dt;
			//incremental_y += cos(incremental_yaw) * rr * dt;
			incremental_x += cos(incremental_yaw) * rr * dt;
			incremental_y += sin(incremental_yaw) * rr * dt;
		} else {
			double w = (rr - rl) / trackwidth;
			double R = trackwidth * 0.5 * (rr + rl) / (rr - rl);
			double icc_x = incremental_x - R * sin(incremental_yaw);
			double icc_y = incremental_y + R * cos(incremental_yaw);

			double wdt = w * dt;
			double new_x = cos(wdt) * (incremental_x - icc_x) - sin(wdt) * (incremental_y - icc_y) + icc_x;
			double new_y = sin(wdt) * (incremental_x - icc_x) + cos(wdt) * (incremental_y - icc_y) + icc_y;
			double new_theta = incremental_yaw + wdt;

			incremental_x = new_x;
			incremental_y = new_y;
			incremental_yaw = new_theta;

		}

	}
	
	/* Update incremental tfm */
	Eigen::AngleAxisd aaZ(incremental_yaw, Eigen::Vector3d::UnitZ());
	//Eigen::AngleAxisd aaY(0, Eigen::Vector3d::UnitY());
	//Eigen::AngleAxisd aaX(0, Eigen::Vector3d::UnitX());
	//Eigen::Quaterniond q = aaZ * aaY * aaX;
	Eigen::Quaterniond q = Eigen::Quaterniond(aaZ);
	incremental_tfm = Sophus::SE3d(q, Eigen::Vector3d(incremental_x, incremental_y, 0));

	//fprintf(differential, "%f %f %f\n", incremental_x, incremental_y, incremental_yaw);
	incremental_timestamp = timestamp;
}

//=============================================================================
void f_gps(double timestamp, double utm_e, double utm_n, double altitude)
{
	gps.push_back( Eigen::Vector3d(utm_e, utm_n, altitude) );

	// world from vehicle
  Sophus::SE3d pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d(utm_e, utm_n, altitude) );
  // Sophus::SE3d start_pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d(utm_e+1, utm_n+0.1, altitude-5) );

  Eigen::Vector3d translation = incremental_pose.translation();
  incremental_pose = incremental_pose * incremental_tfm;

  pose.so3() = incremental_pose.so3();
  nodes.push_back( slam.AddPose(pose , true, timestamp) );

	Eigen::Matrix<double,6,1> cov_diag;
  cov_diag << 3,3,30, DBL_MAX, DBL_MAX, DBL_MAX;

  // cov_diag << 1,1,1,1000,1000,1000;

  pose.so3() = Sophus::SO3();
  slam.AddUnaryConstraint(nodes.back(), pose, cov_diag.asDiagonal());

	if (nodes.size() >= 2)
	{
		unsigned int prev = nodes[nodes.size()-2];
		unsigned int curr = nodes.back();
    slam.AddBinaryConstraint(prev,curr,incremental_tfm);
  //std::cerr << "adding binary constraint between " << prev << std::endl <<
  //             slam.GetPose(prev).t_wp.matrix() << std::endl << " and " <<
  //             curr << std::endl << slam.GetPose(curr).t_wp.matrix() <<
  //             std::endl << " with t = " << std::endl <<
  //             incremental_tfm.matrix() << std::endl;
	}

	fprintf(differential, "%f %f\n", translation[0], translation[1]);
	
	incremental_x = 0;
	incremental_y = 0;
	incremental_yaw = 0;
	incremental_tfm = Sophus::SE3d();
}



//=============================================================================
void setup()
{
	fprintf(stderr, "Init BA\n");
	slam.Init(100,0);

	Eigen::Matrix<double,3,1> gravity;
	gravity << 0,0,9.8;
	slam.SetGravity(gravity);
}

//=============================================================================
void parse_file(const char* filename)
{
	FILE* input = fopen(filename, "r");
	while (1)
	{
		char name[5];
		if (fscanf(input, "%s", name) == EOF) break;
		if (strncmp(name, "ODO", 3) == 0)
		{
			double time, rr, rl;
			if (fscanf(input, "%lf %lf %lf", &time, &rr, &rl) != EOF)
				update_incremental_pose(time, rr,rl);
		} else if (strncmp(name, "UTM", 3) == 0)
		{
			double time, utm_e, utm_n, altitude;
      if (fscanf(input, "%lf %lf %lf %lf", &time, &utm_e, &utm_n, &altitude) != EOF) {
        if (nodes.size() < 10000) {
          f_gps(time, utm_e, utm_n, altitude);
        } else {
          break;
        }
      }
		} else if (strncmp(name, "IMU",3)==0)
		{
			double time;
			double angleRates[3], accels[3];
			if (fscanf(input, "%lf %lf %lf %lf %lf %lf %lf", &time, angleRates, angleRates+1,angleRates+2, accels, accels+1, accels+2) != EOF)
			{}
		}	else {
			fprintf(stderr, "Unknown symbol <%s>\n", name);
		}

	}
	fclose(input);
	//exit(0);
}

//=============================================================================
void solve()
{
	fprintf(stderr, "BA::Solve w [%zu] poses\n", nodes.size());
  slam.Solve(100, 1.0);
	fprintf(stderr, "finish BA::Solve\n");
}

//=============================================================================
int main(int argc, char** argv)
{
	differential = fopen("out","w");

	setup();
	parse_file(argv[1]);
	
	fclose(differential);

  ba::debug_level_threshold = 1;

	solve();
}

