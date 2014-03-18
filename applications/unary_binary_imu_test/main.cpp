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

#include <ba/BundleAdjuster.h>
#include <ba/Types.h>
#include <ba/InterpolationBuffer.h>
#include <Eigen/Eigen>

using std::vector;
using namespace std;

//=============================================================================
ba::BundleAdjuster<double,0,9,0>  slam;

vector<unsigned int> nodes;

double offset_e;
double offset_n;
double offset_u;

vector<Eigen::Vector3d> gps;

/* 2D coordinate transformations */
double incremental_x = 0;
double incremental_y = 0;
double incremental_yaw = 0;
double incremental_timestamp = 0;

Sophus::SE3d incremental_differential_update;
Sophus::SE3d incremental_differential_pose;

double speed = 0;
Sophus::SE3d incremental_gyro_pose;
Sophus::SE3d incremental_gyro_update;

typedef ba::ImuMeasurementT<double>     ImuMeasurement;
ba::InterpolationBufferT<ImuMeasurement, double> imu_buffer;

//=============================================================================
void add_imu(double timestamp, double* angleRates, double* accels)
{
	Eigen::Vector3d w(angleRates[0], angleRates[1], angleRates[2]);
	Eigen::Vector3d a(accels[0], accels[1], accels[2]);

	ImuMeasurement imu(w,a,timestamp);
	imu_buffer.AddElement(imu);
}

//=============================================================================
void add_gyro_and_speed(double timestamp, double xAngleRate, double yAngleRate, double zAngleRate, double speed)
{
	static double last_timestamp = 0;
	if (last_timestamp != 0)
	{
		double dt = timestamp - last_timestamp;
		double distance = speed * dt;

		Eigen::AngleAxisd aaZ(zAngleRate*dt, Eigen::Vector3d::UnitZ());
		Eigen::AngleAxisd aaY(yAngleRate*dt, Eigen::Vector3d::UnitY());
		Eigen::AngleAxisd aaX(xAngleRate*dt, Eigen::Vector3d::UnitX());
		Eigen::Quaterniond q = aaZ * aaY * aaX;
		Sophus::SE3d update(q, Eigen::Vector3d(0,distance,0));
		incremental_gyro_update = incremental_gyro_update * update;
	}
	last_timestamp = timestamp;
}

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

	speed = 0.5 * (rr + rl);

	double dt = timestamp - incremental_timestamp;
	
	const double trackwidth = 1.5;

	double TINY = 0.0001;
	if (fabs(rr) > TINY or fabs(rl) > TINY)
	{
		if (fabs(rr-rl) < TINY)
		{
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
	incremental_differential_update = Sophus::SE3d(q, Eigen::Vector3d(incremental_x, incremental_y, 0));

	incremental_timestamp = timestamp;
}

//=============================================================================

//=============================================================================
void f_gps(double timestamp, double utm_e, double utm_n, double altitude)
{
	static double last_gps_timestamp = 0;
	static bool firsttime = true;
	if (firsttime)
	{
		offset_e = utm_e;
		offset_n = utm_n;
		offset_u = altitude;
		firsttime = false;
	}

	incremental_gyro_pose *= incremental_gyro_update;
	incremental_differential_pose *= incremental_differential_update;
	gps.push_back( Eigen::Vector3d(utm_e, utm_n, altitude) );

  Sophus::SE3d utm_prior(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(utm_e-offset_e, utm_n-offset_n, altitude-offset_u));

	/* First time, use GPS prior at coordinate origin */
	if (nodes.empty())
	{
		Sophus::SE3d pose(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    nodes.push_back( slam.AddPose(pose , true, timestamp) );
	} else {
		unsigned int recent = nodes.back();
		const Sophus::SE3d & recent_pose = slam.GetPose(recent).t_wp;
    Sophus::SE3d estimate = recent_pose * incremental_gyro_update;
    // Sophus::SE3d estimate = recent_pose * incremental_differential_update;
    nodes.push_back( slam.AddPose(estimate, true, timestamp) );
    // std::cerr << "Adding pose at " << estimate.matrix() << std::endl;
	}

	/* Add unary constraint to SLAM */
	{
		Eigen::Matrix<double,6,1> cov_diag;
    cov_diag << 1000,1000,30000, DBL_MAX, DBL_MAX, DBL_MAX;
		slam.AddUnaryConstraint(nodes.back(), utm_prior, cov_diag.asDiagonal());
    // std::cerr << "Adding unary constraint for pose " << nodes.back() << " at " <<
    //              utm_prior.matrix() << std::endl;
	}

	if (nodes.size() >= 2)
	{
    //slam.AddBinaryConstraint(nodes.back()-1, nodes.back(), incremental_differential_update);
    slam.AddBinaryConstraint(nodes.back()-1, nodes.back(), incremental_gyro_update);
	}
	

	if (nodes.size() >= 2)
	{
		vector<ImuMeasurement> imu_meas = imu_buffer.GetRange(last_gps_timestamp, timestamp);
    if (imu_meas.size() == 0) {
      std::cerr << "Could not find imu measurements between : " <<
                   last_gps_timestamp << " and " << timestamp << std::endl;
      exit(0);
    }
    slam.AddImuResidual(nodes.back()-1, nodes.back(), imu_meas);
	}


	incremental_x = 0;
	incremental_y = 0;
	incremental_yaw = 0;
	
	/* reset incremental accumulators */
	incremental_differential_update = Sophus::SE3d();
	incremental_gyro_update = Sophus::SE3d();

	//if (nodes.size() == 50)
	//slam.Solve(100,0.2,false,false);

	last_gps_timestamp = timestamp;
}

//=============================================================================
void setup()
{
	fprintf(stderr, "Init BA\n");
  slam.Init(100,0, 0, Sophus::SE3d(), 100000);

	Eigen::Matrix<double,3,1> gravity;
	gravity << 0,0,9.8;
	slam.SetGravity(gravity);

	/* Initialize incremental gyro pose */
	//Eigen::AngleAxisd aaZ(-M_PI/2.0, Eigen::Vector3d::UnitZ());
	Eigen::AngleAxisd aaZ(0, Eigen::Vector3d::UnitZ());
	Eigen::AngleAxisd aaY(0, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd aaX(0, Eigen::Vector3d::UnitX());
	Eigen::Quaterniond q = aaZ * aaY * aaX;
	incremental_gyro_pose = Sophus::SE3d(q, Eigen::Vector3d::Zero());
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
			{
				add_gyro_and_speed(time, angleRates[0], angleRates[1], angleRates[2], speed);
        add_imu(time, angleRates, accels);
			}
		}	else {
			fprintf(stderr, "Unknown symbol <%s>\n", name);
		}

	}
	fclose(input);
}

//=============================================================================
void solve()
{
	fprintf(stderr, "BA::Solve w [%zu] poses\n", nodes.size());
  slam.Solve(25, 0.2, false, true);
	fprintf(stderr, "finish BA::Solve\n");
}

//=============================================================================
int main(int argc, char** argv)
{
	setup();
	parse_file(argv[1]);
	
  ba::debug_level_threshold = 0;

	solve();
}

