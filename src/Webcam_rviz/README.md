# Webcam_Rviz


### install the ROS usb_cam
```bash
sudo apt-get install ros-noetic-usb-cam
```

```bash
mkdir cat22kin_ws && cd catkin_ws && mkdir src && cd src

git clone https://github.com/Ahmed-Magdi1/Webcam_rviz.git

cd ../

catkin_make

source devel/setup.bash 
```
### In "cam.launch" adjust the camera port default to `/dev/video0`
``<param name="video_device" value="<camera_port>" />``

### Run the launch file
```
roslaunch cam cam.launch 
```

### To see the camera in RViz. Open new Terminal
```bash
rosrun rviz rviz
```
then add the camera RViz and add the camera topic "/usb_cam/image_raw"


