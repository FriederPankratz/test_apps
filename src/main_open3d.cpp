#include <iostream>

#include <traact/traact.h>

#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <traact/serialization/JsonGraphInstance.h>

#include <traact/util/Logging.h>
#include <signal.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <traact/vision.h>
#include <traact/point_cloud.h>

using namespace traact;
using namespace traact::dataflow;

const static std::string kSingleWindow{"window"};
const static std::string kSceneWindow{"scene"};

const static std::string kImage{"image"};
const static std::string kPointCloudVertex{"pointCloud_vertex"};
const static std::string kPointCloudColor{"pointCloud_color"};
const static std::string kPointCloudPose{"pose_origin"};

const static std::string kMarkerPose{"pose_marker_{0}_pose"};

const static std::string kDebugCalibration{"calibration"};
const static std::string kDebugPose{"pose_{0}"};

const static int countMarker = 1;
const static int originMarker = 0;
const static int mainCamera = 0;
const static double marker_size = 0.131;


std::string getRawAppPortName(std::string render_in, int object_id, std::string purpose) {
    return fmt::format("{0}_{1}_{2}", render_in, object_id, purpose);
}

std::string getRawAppPortName(std::string render_in, std::string object_id, std::string purpose) {
    return fmt::format("{0}_{1}_{2}", render_in, object_id, purpose);
}

std::string getPoseAppPortName(std::string render_in, std::string object_id, std::string origin) {
    return getRawAppPortName(render_in, object_id, fmt::format("pose_{0}", origin));
}

std::string getCalibAppPortName(std::string render_in, std::string object_id, std::string origin) {
    return getRawAppPortName(render_in, object_id, fmt::format("calibration_{0}", origin));
}

std::string getCalibReaderComponentName(std::string source, std::string target) {
    return fmt::format("scene_{1}_calibrationRead_{0}", source, target);
}

std::string getCalibWriterComponentName(std::string source, std::string target) {
    return fmt::format("scene_{1}_calibrationWrite_{0}", source, target);
}

std::string getStaticPoseComponentName(std::string source, std::string target) {
    return fmt::format("scene_{1}_static_{0}", source, target);
}

void calcAlphaBeta(double threshold, double min, double max, double &alpha, double &beta) {
    alpha = 255.0 / (max - min);
    beta = -255.0;
}

bool running = true;
traact::facade::DefaultFacade my_facade;

void addCalibProcessing(DefaultInstanceGraphPtr graph, int camera_index, std::string world_to_camera_file);
void ctrlC(int i) {
    SPDLOG_INFO("User requested exit (Ctrl-C).");
    my_facade.stop();
}

std::string getIdxName(std::string name, int idx) {
    return fmt::format("{0}_{1}", name, idx);
}

std::string getFileName(std::string base_path, std::string name) {
    return fmt::format(base_path, name);
}

std::string getCameraFileName(std::string base_path, std::string name, int idx) {
    return fmt::format(base_path, name, idx);
}



void addDebugView(traact::DefaultInstanceGraphPtr &graph, int camera_index) {
    auto get_name = [camera_index](const std::string &name) {
        return getIdxName(name, camera_index);
    };


    graph->connect(get_name("download_color"), "output", "debug_sink", getRawAppPortName(kSingleWindow, camera_index, kImage));
    graph->connect(get_name("create_point_cloud"), "output", "debug_sink", getRawAppPortName(kSceneWindow, camera_index, kPointCloudVertex));
    graph->connect(get_name("color_point_cloud"), "output", "debug_sink", getRawAppPortName(kSceneWindow, camera_index, kPointCloudColor));

    graph->connect(get_name("depth_to_color"), "output", "debug_sink", getPoseAppPortName(kSceneWindow, fmt::format("color{0}", camera_index), std::to_string(camera_index)));

    graph->connect(get_name("undistort_color"), "output_calibration", "debug_sink", getRawAppPortName(kSingleWindow, camera_index, kDebugCalibration));

    //graph->connect(getCalibComponentName("origin", std::to_string(index)), "output", "debug_sink", getCalibAppPortName(kSceneWindow, std::to_string(camera_index), "origin"));




}



void addBasicProcessing(traact::DefaultInstanceGraphPtr &graph,
                        int index,
                        std::string filename,
                        std::string world_to_camera_file,
                        pattern::instance::PatternInstance::Ptr &register_icp_pattern) {
    auto get_name = [index](const std::string &name) {
        return getIdxName(name, index);
    };

    auto source_pattern =
        graph->addPattern(get_name("source"),
                          my_facade.instantiatePattern("traact::component::kinect::KinectAzureSingleFilePlayer"));

    auto upload_depth_pattern =
        graph->addPattern(get_name("upload_depth"), my_facade.instantiatePattern("OpenCvCudaUpload"));
    auto upload_xy_table_pattern =
        graph->addPattern(get_name("upload_xy_table"), my_facade.instantiatePattern("OpenCvCudaUpload"));
    auto upload_color_pattern =
        graph->addPattern(get_name("upload_color"), my_facade.instantiatePattern("OpenCvCudaUpload"));
    auto create_point_cloud_pattern =
        graph->addPattern(get_name("create_point_cloud"), my_facade.instantiatePattern("CudaCreatePointCloud"));
    auto color_point_cloud_pattern =
        graph->addPattern(get_name("color_point_cloud"), my_facade.instantiatePattern("CudaColorPointCloud"));


    auto undistort_color_pattern =
        graph->addPattern(get_name("undistort_color"), my_facade.instantiatePattern("OpenCvCudaUndistortImage"));

    auto color_to_gray_pattern =
        graph->addPattern(get_name("color_to_gray"), my_facade.instantiatePattern("OpenCvCudaCvtColor"));
    auto download_gray_pattern =
        graph->addPattern(get_name("download_gray"), my_facade.instantiatePattern("OpenCvCudaDownload"));
    auto download_color_pattern =
        graph->addPattern(get_name("download_color"), my_facade.instantiatePattern("OpenCvCudaDownload"));
    auto origin_tracker_pattern =
        graph->addPattern(get_name("origin_tracker"), my_facade.instantiatePattern("ApriltagTracker"));

    auto depth_to_color_pattern =
        graph->addPattern(get_name("depth_to_color"), my_facade.instantiatePattern("InversionPose6D"));


    auto origin_to_camera_pattern =
        graph->addPattern(getCalibReaderComponentName("origin", std::to_string(index)), my_facade.instantiatePattern("FileReaderWriterRead_cereal_traact::spatial::Pose6D"));
    auto origin_to_camera_write_pattern =
        graph->addPattern(getCalibWriterComponentName("origin", std::to_string(index)), my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));


    auto origin_to_camera_color_mul_pattern =
        graph->addPattern(get_name("origin_to_color_camera_mul"), my_facade.instantiatePattern("MultiplicationPose6DPose6D"));
    auto origin_to_camera_depth_mul_pattern =
        graph->addPattern(get_name("origin_to_depth_camera_mul"), my_facade.instantiatePattern("MultiplicationPose6DPose6D"));

    auto marker_to_camera_pattern =
        graph->addPattern(get_name("marker_to_camera"), my_facade.instantiatePattern("InversionPose6D"));

    auto gate_origin_to_camera_pattern =
        graph->addPattern(get_name("gate_origin_to_camera"), my_facade.instantiatePattern("Gate_traact::spatial::Pose6D"));

    auto gate_point_cloud_pattern =
        graph->addPattern(get_name("gate_point_cloud"), my_facade.instantiatePattern("Gate_traact::vision::GpuImage"));
    auto gate_point_color_pattern =
        graph->addPattern(get_name("gate_point_color"), my_facade.instantiatePattern("Gate_traact::vision::GpuImage"));

    auto download_point_image_pattern =
        graph->addPattern(get_name("download_point_cloud"), my_facade.instantiatePattern("OpenCvCudaDownload"));
    auto download_point_color_pattern =
        graph->addPattern(get_name("download_point_color"), my_facade.instantiatePattern("OpenCvCudaDownload"));
    auto build_point_cloud_pattern =
        graph->addPattern(get_name("build_point_cloud"), my_facade.instantiatePattern("Open3DBuildPointCloud"));



    // configure
    source_pattern->setParameter("file", filename);
    source_pattern->setParameter("stop_after_n_frames", -1);
    source_pattern->setParameter("send_same_frame_as_new_after_stop", true);


    //origin_tracker_pattern->setParameter("Dictionary", "DICT_4X4_50");
    //origin_tracker_pattern->setParameter("marker_size", marker_size);

    origin_to_camera_pattern->setParameter("file", world_to_camera_file);
    origin_to_camera_pattern->setParameter("CoordinateSystem", "OpenGL");
    origin_to_camera_write_pattern->setParameter("file", world_to_camera_file);
    origin_to_camera_write_pattern->setParameter("CoordinateSystem", "OpenGL");



    download_point_image_pattern->setParameter("cuda_graph", "download_for_icp");
    download_point_color_pattern->setParameter("cuda_graph", "download_for_icp");



    for(int i=0;i<countMarker;++i){
        auto& marker = origin_tracker_pattern->instantiatePortGroup("output_pose");
        marker.setParameter("marker_id", i);
        marker.setParameter("marker_size", marker_size);
        graph->connect(get_name("origin_tracker"), marker.getProducerPortName("output"), "debug_sink", getPoseAppPortName(kSceneWindow, fmt::format("cam{0}marker{1}",index, i), fmt::format("color{0}", index)));

        graph->connect(get_name("origin_tracker"), marker.getProducerPortName("output"), "debug_sink", getRawAppPortName(kSingleWindow, index, fmt::format(kDebugPose, i)));

        if(i == originMarker){
            graph->connect(get_name("origin_tracker"), marker.getProducerPortName("output"), get_name("marker_to_camera"), "input");
        }

    }

    // setup connections
    graph->connect(get_name("source"), "output_depth", get_name("upload_depth"), "input");
    graph->connect(get_name("source"), "output_xy_table", get_name("upload_xy_table"), "input");
    graph->connect(get_name("source"), "output", get_name("upload_color"), "input");

    graph->connect(get_name("upload_depth"), "output", get_name("create_point_cloud"), "input");
    graph->connect(get_name("upload_xy_table"), "output", get_name("create_point_cloud"), "input_xy_table");

    graph->connect(get_name("create_point_cloud"), "output", get_name("color_point_cloud"), "input");
    graph->connect(get_name("upload_color"), "output", get_name("color_point_cloud"), "input_color");
    graph->connect(get_name("source"), "output_calibration", get_name("color_point_cloud"), "input_color_calibration");
    graph->connect(get_name("source"), "output_color_to_depth", get_name("color_point_cloud"), "input_color_to_depth");

    graph->connect(get_name("upload_color"), "output", get_name("undistort_color"), "input");
    graph->connect(get_name("source"), "output_calibration", get_name("undistort_color"), "input_calibration");
    graph->connect(get_name("undistort_color"), "output", get_name("color_to_gray"), "input");
    graph->connect(get_name("color_to_gray"), "output", get_name("download_gray"), "input");
    graph->connect(get_name("undistort_color"), "output", get_name("download_color"), "input");

    graph->connect(get_name("download_gray"), "output", get_name("origin_tracker"), "input");
    graph->connect(get_name("undistort_color"), "output_calibration", get_name("origin_tracker"), "input_calibration");
    graph->connect(get_name("source"), "output_color_to_depth", get_name("depth_to_color"), "input");


    graph->connect(getCalibReaderComponentName("origin", "marker"), "output", get_name("origin_to_color_camera_mul"), "input_a");
    graph->connect(get_name("marker_to_camera"), "output", get_name("origin_to_color_camera_mul"), "input_b");

    graph->connect(get_name("origin_to_color_camera_mul"), "output", get_name("origin_to_depth_camera_mul"), "input_a");
    graph->connect(get_name("source"), "output_color_to_depth", get_name("origin_to_depth_camera_mul"), "input_b");


    graph->connect(get_name("origin_to_depth_camera_mul"), "output", get_name("gate_origin_to_camera"), "input");
    graph->connect("register_using_marker", "output", get_name("gate_origin_to_camera"), "input_event");
    graph->connect(get_name("gate_origin_to_camera"), "output", getCalibWriterComponentName("origin", std::to_string(index)), "input");



    graph->connect(get_name("create_point_cloud"), "output", get_name("gate_point_cloud"), "input");
    graph->connect("register_using_icp", "output", get_name("gate_point_cloud"), "input_event");
//
    graph->connect(get_name("color_point_cloud"), "output", get_name("gate_point_color"), "input");
    graph->connect("register_using_icp", "output", get_name("gate_point_color"), "input_event");
//
    graph->connect(get_name("gate_point_cloud"), "output", get_name("download_point_cloud"), "input");
    graph->connect(get_name("gate_point_color"), "output", get_name("download_point_color"), "input");
//
    graph->connect(get_name("download_point_cloud"), "output", get_name("build_point_cloud"), "input");
    graph->connect(get_name("download_point_color"), "output", get_name("build_point_cloud"), "input_color");


    auto icp_camera = register_icp_pattern->instantiatePortGroup("camera");
    auto origin_to_camera_write_icp_pattern =
        graph->addPattern(get_name("origin_to_camera_write_icp"), my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));
    origin_to_camera_write_icp_pattern->setParameter("file", world_to_camera_file);
    origin_to_camera_write_icp_pattern->setParameter("CoordinateSystem", "OpenGL");

    graph->connect(getCalibReaderComponentName("origin", std::to_string(index)), "output", "register_icp", icp_camera.getConsumerPortName("input_pose"));
    graph->connect(get_name("build_point_cloud"), "output", "register_icp", icp_camera.getConsumerPortName("input_cloud"));
    graph->connect("register_icp", icp_camera.getProducerPortName("output"), get_name("origin_to_camera_write_icp"), "input");


}
int main(int argc, char **argv) {

    using namespace traact;
    using namespace traact::dataflow;
    using namespace traact::facade;

    signal(SIGINT, ctrlC);

    util::initLogging(spdlog::level::debug);

    DefaultInstanceGraphPtr graph = std::make_shared<DefaultInstanceGraph>("point_cloud_from_mkv_multiway");

    std::vector<int> camera_dirs{1,2,3,4,5};
    int camera_count = camera_dirs.size();
    std::string video_pattern = "/artekmed/recordings/calib/cn{0:02d}/capture_cn{0:02d}.mkv";
    std::string origin_to_camera_pattern = "/artekmed/config/calibration/cn{0:02d}/world2camera.json";
    std::string origin_to_marker_file = "/artekmed/config/calibration/origin_to_marker.json";

    // prepare a RawApplicationSyncSink with all ports used for debug rendering
    auto debug_pattern = my_facade.instantiatePattern("RawApplicationSyncSink");
    for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
        debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow, camera_index, kImage), traact::vision::ImageHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow, camera_index, kDebugCalibration), traact::vision::CameraCalibrationHeader::NativeTypeName);
        //debug_pattern->addConsumerPort(getRawAppPortName(kDebugPointCloud, camera_index), traact::point_cloud::PointCloudHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kSceneWindow, camera_index, kPointCloudVertex), traact::vision::GpuImageHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kSceneWindow, camera_index, kPointCloudColor), traact::vision::GpuImageHeader::NativeTypeName);

        //debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow, std::to_string(camera_index), "origin"), traact::spatial::Pose6DHeader::NativeTypeName);

        debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow, fmt::format("color{0}", camera_index), std::to_string(camera_index)), traact::spatial::Pose6DHeader::NativeTypeName);




        for(int i=0;i<countMarker;++i){
            debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow, fmt::format("cam{0}marker{1}",camera_index, i), fmt::format("color{0}", camera_index)), traact::spatial::Pose6DHeader::NativeTypeName);
            debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow, camera_index, fmt::format(kDebugPose, i)), traact::spatial::Pose6DHeader::NativeTypeName);
        }



    }

    graph->addPattern("debug_sink", debug_pattern);

//    auto origin_to_marker_pattern =
//        graph->addPattern(getStaticPoseComponentName("origin", "marker"), my_facade.instantiatePattern("StaticPose"));
//    origin_to_marker_pattern->setParameter("rx", -0.707107);
//    origin_to_marker_pattern->setParameter("ry", 0.0);
//    origin_to_marker_pattern->setParameter("rz", 0.0);
//    origin_to_marker_pattern->setParameter("rw", 0.707107);

    auto origin_to_marker_pattern =
        graph->addPattern(getCalibReaderComponentName("origin", "marker"), my_facade.instantiatePattern("FileReaderWriterRead_cereal_traact::spatial::Pose6D"));
    auto origin_to_marker_write_pattern =
        graph->addPattern(getCalibWriterComponentName("origin", "marker"), my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));
    origin_to_marker_pattern->setParameter("file", origin_to_marker_file);
    origin_to_marker_pattern->setParameter("CoordinateSystem", "OpenGL");
    origin_to_marker_write_pattern->setParameter("file", origin_to_marker_file);
    origin_to_marker_write_pattern->setParameter("CoordinateSystem", "OpenGL");
    graph->connect(getCalibReaderComponentName("origin", "marker"), "output", getCalibWriterComponentName("origin", "marker"), "input");


    auto register_using_marker_pattern =
        graph->addPattern("register_using_marker", my_facade.instantiatePattern("SyncUserEvent"));
    auto register_using_icp_pattern =
        graph->addPattern("register_using_icp", my_facade.instantiatePattern("SyncUserEvent"));

    //pattern::instance::PatternInstance::Ptr register_icp_pattern = graph->addPattern("register_icp", my_facade.instantiatePattern("Open3DMultiCameraColorICP"));
    pattern::instance::PatternInstance::Ptr register_icp_pattern = graph->addPattern("register_icp", my_facade.instantiatePattern("Open3DMultiwayRegistration"));
    register_icp_pattern->setParameter("reference_node", mainCamera);







    for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
        auto video_file =
            fmt::format(video_pattern, camera_dirs[camera_index]);
        auto world_to_camera_file =
            fmt::format(origin_to_camera_pattern, camera_dirs[camera_index]);
        addBasicProcessing(graph, camera_index, video_file, world_to_camera_file, register_icp_pattern);
        addDebugView(graph, camera_index);

    }
    //addInitTracking(graph, camera_count, init_file_pattern);
    //addPointEstimation(graph, camera_count, init_file_pattern);
    //addOutsideInTargetTracking(graph, camera_count);
    //addOutsideInTargetRecord(graph, camera_count, result_file_pattern, result_camera_file_pattern);

    buffer::TimeDomainManagerConfig td_config;
    td_config.time_domain = 0;
    td_config.ringbuffer_size = 3;

    td_config.source_mode = SourceMode::WAIT_FOR_BUFFER;
    td_config.missing_source_event_mode = MissingSourceEventMode::WAIT_FOR_EVENT;
    td_config.max_offset = std::chrono::milliseconds(10);
    td_config.max_delay = std::chrono::milliseconds(100);
    td_config.sensor_frequency = 30;
    td_config.cpu_count = 0;

    graph->timedomain_configs[0] = td_config;



    std::string filename = graph->name + ".json";
    {
        nlohmann::json jsongraph;
        ns::to_json(jsongraph, *graph);

        std::ofstream myfile;
        myfile.open(filename);
        myfile << jsongraph.dump(4);
        myfile.close();

        std::cout << jsongraph.dump(4) << std::endl;
    }

    //my_facade.loadDataflow(graph);
//
    //my_facade.blockingStart();

    SPDLOG_INFO("exit program");

    //SPDLOG_INFO("run the same dataflow again using: traactConsole {0}", filename);

    return 0;
}

