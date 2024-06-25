Test apps that include many components and generate dataflows

copy imgui.ini into you execution directory (e.g. build/Debug)

# ICP Registration

1. edit misc/registration_config.yml
2. activate conan run env
3. run: ./open3d_generator {PATH_TO}/registration_config.yml
4. two files will be generated, registration_icp_shm.json with shm form pcpd as input and registration_icp_mkv.json with k4a_recorder input 
5. run: traact_gui registration_icp_shm.json

## Parameter registration_config.yml

- name: base name of generated files
- main_camera: id used for marker tracking  
- origin_marker_id: id of origin marker
- origin_to_camera_file_pattern: format string defining the result file name. {0:02d} is replaced by camera id
- origin_to_marker_file: path to cereal json pose file. **must exist**. can be modified in traact_gui (e.g. misc/identity_pose.json)
- add_single_windows: add windows with rgb image and debug rendering for every camera 
- cameras: list of camera ids
- marker_tracker: section named marker_tracker is used for tracking. Default is Apriltag, examples for Aruco and ArucoFractal are included 
- video_file_pattern: like origin_to_camera_file_pattern, defines path to video input files
- image_stream: name of shm getImage for video input


