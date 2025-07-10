#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <ros/console.h>
#include <string>

class TeleopJoy
{
public:
    TeleopJoy();

private:
    void joyCallback(const sensor_msgs::Joy::ConstPtr &joy);

    ros::NodeHandle nh_;

    int linear_x_{1}, linear_y_{1}, angular_z_{0};
    int linear_x_2{3}, linear_y_2{3}, angular_z_2{2};
    double lin_x_slow_{0.1}, lin_y_slow_{0.1}, ang_z_slow_{0.5};
    double lin_x_normal_{0.3}, lin_y_normal_{0.3}, ang_z_normal_{1.0};
    double lin_x_fast_{0.7}, lin_y_fast_{0.7}, ang_z_fast_{1.0};

    ros::Publisher vel_pub_;
    ros::Subscriber joy_sub_;
};

TeleopJoy::TeleopJoy()
{
    nh_.param("axis_linear_x", linear_x_, linear_x_);
    nh_.param("axis_linear_y", linear_y_, linear_y_);
    nh_.param("axis_angular_z", angular_z_, angular_z_);

    nh_.param("slow_linear_x",  lin_x_slow_, lin_x_slow_);
    nh_.param("slow_linear_y",  lin_y_slow_, lin_y_slow_);
    nh_.param("slow_angular_z", ang_z_slow_, ang_z_slow_);

    nh_.param("normal_linear_x",  lin_x_normal_, lin_x_normal_);
    nh_.param("normal_linear_y",  lin_y_normal_, lin_y_normal_);
    nh_.param("normal_angular_z", ang_z_normal_, ang_z_normal_);

    nh_.param("fast_linear_x",  lin_x_fast_, lin_x_fast_);
    nh_.param("fast_linear_y",  lin_y_fast_, lin_y_fast_);
    nh_.param("fast_angular_z", ang_z_fast_, ang_z_fast_);

    vel_pub_ = nh_.advertise<geometry_msgs::Twist>("joy_vel", 1);
    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 1, &TeleopJoy::joyCallback, this);
}

void TeleopJoy::joyCallback(const sensor_msgs::Joy::ConstPtr &joy)
{
    geometry_msgs::Twist twist;

   // if (joy->buttons[4] == 1)
   // {
        if (joy->buttons[0] == 1)
        {
            //slow
            twist.linear.x = lin_x_slow_ * joy->axes[linear_x_];
            twist.linear.y = lin_y_slow_ * joy->axes[linear_y_];
            twist.angular.z = ang_z_slow_ * joy->axes[angular_z_];
            //twist.linear.x = lin_x_slow_ * joy->axes[linear_x_2];
            //twist.linear.y = lin_y_slow_ * joy->axes[linear_y_2];
           // twist.angular.z = ang_z_slow_ * joy->axes[angular_z_2];
        }
        else if (joy->buttons[3] == 1)
        {
            //fast
            twist.linear.x = lin_x_normal_ * joy->axes[linear_x_];
            twist.linear.y = lin_y_normal_ * joy->axes[linear_y_];
            twist.angular.z = ang_z_normal_ * joy->axes[angular_z_];
            //twist.linear.x = lin_x_normal_ * joy->axes[linear_x_2];
            //twist.linear.y = lin_y_normal_ * joy->axes[linear_y_2];
           // twist.angular.z = ang_z_normal_ * joy->axes[angular_z_2];
        }
        else if (joy->buttons[2] == 1)
        {
            //normal
            twist.linear.x = lin_x_fast_ * joy->axes[linear_x_];
            twist.linear.y = lin_y_fast_ * joy->axes[linear_y_];
            twist.angular.z = ang_z_fast_ * joy->axes[angular_z_];
            //twist.linear.x = lin_x_fast_ * joy->axes[linear_x_2];
            //twist.linear.y = lin_y_fast_ * joy->axes[linear_y_2];
            //twist.angular.z = ang_z_fast_ * joy->axes[angular_z_2];
        }
        else
    {
        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = 0.0;
    }
    //}
    /*else
    {
        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = 0.0;
    }*/
    
    vel_pub_.publish(twist);
}

int main(int argc, char **argv)
{
    ROS_INFO("Init Node");
    ros::init(argc, argv, "f710_teleop_joy_node");
    TeleopJoy teleop_joy;
    ros::spin();
}
