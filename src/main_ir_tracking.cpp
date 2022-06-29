#include <iostream>

#include <traact/traact.h>

#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <traact/serialization/JsonGraphInstance.h>

#include <traact/util/Logging.h>
#include <signal.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <traact/vision.h>

using namespace traact;
using namespace traact::dataflow;

const static std::string kDebugImage{"image"};
const static std::string kDebugPoint2D{"point2D"};
const static std::string kDebugPoint3D{"point3D"};
const static std::string kDebugCalibration{"calibration"};
const static std::string kDebugPose6D{"pose6D"};
std::string getRawAppPortName(std::string name, int idx) {
    return fmt::format("{0}_{1}", idx, name);
}

void calcAlphaBeta(double threshold, double min, double max, double &alpha, double &beta) {
    alpha = 255.0 / (max - min);
    beta = -255.0;
}

bool running = true;
traact::facade::DefaultFacade my_facade;

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

void addTarget(traact::DefaultInstanceGraphPtr &graph) {
    auto target_model_pattern = graph->addPattern("target_model",
                                                  my_facade.instantiatePattern(
                                                      "FileReader_cereal_traact::vision::Position3DList"));
    target_model_pattern->setParameter("file",
                                       "/home/frieder/projects/traact_workspace/test_apps/misc/BoardTarget.json");
}

void addOutsideInTargetRecord(traact::DefaultInstanceGraphPtr &graph,
                              int camera_count,
                              const std::string &result_base_path, const std::string &result_camera_base_path) {

    auto target_pose_recorder_pattern = graph->addPattern("target_pose_recorder",
                                                          my_facade.instantiatePattern(
                                                              "FileRecorder_cereal_traact::spatial::Pose6D"));
    auto target_feature_recorder_pattern = graph->addPattern("target_feature_recorder",
                                                             my_facade.instantiatePattern(
                                                                 "FileRecorder_cereal_traact::vision::FeatureList"));
    auto point_3D_feature_recorder_pattern = graph->addPattern("point_3D_feature_recorder",
                                                            my_facade.instantiatePattern(
                                                                "FileRecorder_cereal_traact::vision::FeatureList"));

    target_pose_recorder_pattern->setParameter("file", getFileName(result_base_path, "target_pose"));
    target_feature_recorder_pattern->setParameter("file", getFileName(result_base_path, "target_feature"));
    point_3D_feature_recorder_pattern->setParameter("file", getFileName(result_base_path, "point_3D_feature"));

    graph->connect("track_target", "output", "target_pose_recorder", "input");
    graph->connect("find_target", "output", "target_feature_recorder", "input");
    graph->connect("point_estimation", "output_feature", "point_3D_feature_recorder", "input");

    for (int index = 0; index < camera_count; ++index) {
        auto get_name = [index](const std::string &name) {
            return getIdxName(name, index);
        };

        auto calibration_writer_pattern =
            graph->addPattern(get_name("calibration_writer"),
                              my_facade.instantiatePattern("FileWriter_cereal_traact::vision::CameraCalibration"));
        auto point_2D_recorder_pattern =
            graph->addPattern(get_name("point_2D_recorder"),
                              my_facade.instantiatePattern("FileRecorder_cereal_traact::vision::KeyPointList"));
        auto point_2D_feature_recorder_pattern =
            graph->addPattern(get_name("point_2D_feature_recorder"),
                              my_facade.instantiatePattern("FileRecorder_cereal_traact::vision::FeatureList"));

        graph->connect(get_name("undistort"), "output_calibration", get_name("calibration_writer"), "input");
        graph->connect(get_name("blob_detection"), "output", get_name("point_2D_recorder"), "input");
        graph->connect(get_name("blob_detection"), "output_feature", get_name("point_2D_feature_recorder"), "input");

        calibration_writer_pattern->setParameter("file", getCameraFileName(result_camera_base_path, "calibration", index));
        point_2D_recorder_pattern->setParameter("file", getCameraFileName(result_camera_base_path, "point_2D", index));
        point_2D_feature_recorder_pattern->setParameter("file", getCameraFileName(result_camera_base_path, "point_2D_feature", index));

    }

}

void addOutsideInTargetTracking(traact::DefaultInstanceGraphPtr &graph, int camera_count) {

    addTarget(graph);
    auto find_target_pattern =
        graph->addPattern("find_target", my_facade.instantiatePattern("FindTargetInPosition3DList"));
    auto track_target_pattern =
        graph->addPattern("track_target", my_facade.instantiatePattern("OutsideInPoseEstimation"));

    graph->connect("point_estimation", "output", "find_target", "input");
    graph->connect("point_estimation", "output_feature", "find_target", "input_feature");
    graph->connect("target_model", "output", "find_target", "input_model");

    graph->connect("target_model", "output", "track_target", "input");
    graph->connect("find_target", "output", "track_target", "input_feature");
    graph->connect("point_estimation", "output_feature", "track_target", "input_points_feature");

    for (int index = 0; index < camera_count; ++index) {
        auto get_name = [index](const std::string &name) {
            return getIdxName(name, index);
        };

        auto pose_multiplication_pattern =
            graph->addPattern(get_name("pose_multiplication"),
                              my_facade.instantiatePattern("MultiplicationPose6DPose6D"));

        auto &track_target_group = track_target_pattern->instantiatePortGroup("camera");

        graph->connect(get_name("read_camera2world"), "output",
                       "track_target", track_target_group.getConsumerPortName("pose"));
        graph->connect(get_name("undistort"), "output_calibration",
                       "track_target", track_target_group.getConsumerPortName("calibration"));
        graph->connect(get_name("blob_detection"), "output",
                       "track_target", track_target_group.getConsumerPortName("points"));
        graph->connect(get_name("blob_detection"), "output_feature",
                       "track_target", track_target_group.getConsumerPortName("points_feature"));

        graph->connect(get_name("read_camera2world"), "output", get_name("pose_multiplication"), "input_a");
        graph->connect("track_target", "output", get_name("pose_multiplication"), "input_b");
        graph->connect(get_name("pose_multiplication"), "output", "debug_sink", getRawAppPortName(kDebugPose6D, index));

    }

}

void addPointEstimation(traact::DefaultInstanceGraphPtr &graph,
                        int camera_count,
                        const std::string camera2world_filename) {
    auto point_estimation_pattern =
        graph->addPattern("point_estimation", my_facade.instantiatePattern("OutsideInPointEstimation"));

    for (int index = 0; index < camera_count; ++index) {
        auto get_name = [index](const std::string &name) {
            return getIdxName(name, index);
        };

        auto camera_2_world_pattern = graph->addPattern(get_name("read_camera2world"),
                                                        my_facade.instantiatePattern(
                                                            "FileReader_cereal_traact::spatial::Pose6D"));
        auto point_multiplication_pattern =
            graph->addPattern(get_name("point_multiplication"),
                              my_facade.instantiatePattern("MultiplicationPose6DPosition3DList"));

        camera_2_world_pattern->setParameter("file", fmt::format(camera2world_filename, index + 1));

        auto &new_port_group = point_estimation_pattern->instantiatePortGroup("camera");

        graph->connect(get_name("read_camera2world"), "output",
                       "point_estimation", new_port_group.getConsumerPortName("pose"));
        graph->connect(get_name("undistort"), "output_calibration",
                       "point_estimation", new_port_group.getConsumerPortName("calibration"));
        graph->connect(get_name("blob_detection"), "output",
                       "point_estimation", new_port_group.getConsumerPortName("points"));
        graph->connect(get_name("blob_detection"), "output_feature",
                       "point_estimation", new_port_group.getConsumerPortName("points_feature"));

        graph->connect(get_name("read_camera2world"), "output", get_name("point_multiplication"), "input_a");
        graph->connect("point_estimation", "output", get_name("point_multiplication"), "input_b");
        graph->connect(get_name("point_multiplication"), "output",  "debug_sink", getRawAppPortName(kDebugPoint3D, index));



    }

}

void addInitTracking(traact::DefaultInstanceGraphPtr &graph, int camera_count, const std::string &result_file_pattern) {
    auto gather_init_pattern = graph->addPattern("gather_init", my_facade.instantiatePattern("GatherCameraInitPose"));

    addTarget(graph);

    for (int index = 0; index < camera_count; ++index) {
        auto get_name = [index](const std::string &name) {
            return getIdxName(name, index);
        };
        auto estimate_pose_pattern =
            graph->addPattern(get_name("estimate_pose"), my_facade.instantiatePattern("BruteForceEstimatePose"));
        auto render_init_pose_pattern =
            graph->addPattern(get_name("render_init_pose"), my_facade.instantiatePattern("RenderPose6D"));
        auto write_init_pattern = graph->addPattern(get_name("write_init"),
                                                    my_facade.instantiatePattern(
                                                        "FileWriter_cereal_traact::spatial::Pose6D"));
        auto render_pose_pattern =
            graph->addPattern(get_name("render_pose"), my_facade.instantiatePattern("RenderPose6D"));

        estimate_pose_pattern->setParameter("maxPointDistance", 150.0);
        estimate_pose_pattern->setParameter("minError", 0.5);
        estimate_pose_pattern->setParameter("maxError", 1.0);
        render_pose_pattern->setParameter("Window", get_name("Camera"));

        graph->connect(get_name("blob_detection"), "output", get_name("estimate_pose"), "input");
        graph->connect(get_name("undistort"), "output_calibration", get_name("estimate_pose"), "input_calibration");
        graph->connect("target_model", "output", get_name("estimate_pose"), "input_model");

        render_init_pose_pattern->setParameter("Window", get_name("Camera"));
        write_init_pattern->setParameter("file", fmt::format(result_file_pattern, index + 1));

        auto &new_port_group = gather_init_pattern->instantiatePortGroup("Camera");
        graph->connect(get_name("estimate_pose"), "output",
                       gather_init_pattern->instance_id, new_port_group.getConsumerPortName("input"));

        graph->connect(gather_init_pattern->instance_id, new_port_group.getProducerPortName("output"),
                       get_name("write_init"), "input");

        graph->connect(gather_init_pattern->instance_id, new_port_group.getProducerPortName("output"),
                       get_name("render_init_pose"), "input");
        graph->connect(get_name("undistort"), "output_calibration", get_name("render_init_pose"), "input_calibration");
        graph->connect(get_name("estimate_pose"), "output", get_name("render_pose"), "input");
        graph->connect(get_name("undistort"), "output_calibration", get_name("render_pose"), "input_calibration");

    }

}

void addDebugView(traact::DefaultInstanceGraphPtr &graph, int index) {
    auto get_name = [index](const std::string &name) {
        return getIdxName(name, index);
    };


    graph->connect(get_name("undistort"), "output", "debug_sink", getRawAppPortName(kDebugImage, index));
    graph->connect(get_name("undistort"), "output_calibration", "debug_sink", getRawAppPortName(kDebugCalibration, index));
    graph->connect(get_name("blob_detection"), "output", "debug_sink", getRawAppPortName(kDebugPoint2D, index));

}

void addBasicProcessing(traact::DefaultInstanceGraphPtr &graph, int index, std::string filename) {
    auto get_name = [index](const std::string &name) {
        return getIdxName(name, index);
    };

    auto source_pattern =
        graph->addPattern(get_name("source"),
                          my_facade.instantiatePattern("traact::component::kinect::KinectAzureSingleFilePlayer"));

    auto convert_to_gray_pattern =
        graph->addPattern(get_name("convert_to_gray"), my_facade.instantiatePattern("OpenCvConvertImage"));
    auto undistort_pattern =
        graph->addPattern(get_name("undistort"), my_facade.instantiatePattern("OpenCvUndistortImage"));

    auto blob_detection_pattern =
        graph->addPattern(get_name("blob_detection"), my_facade.instantiatePattern("OpenCvBlobDetection"));

    //auto refine_circles_pattern = graph->addPattern(getName("refine_circles"), my_facade.instantiatePattern("RefineCircles"));



    // configure
    source_pattern->setParameter("file", filename);
    source_pattern->setParameter("stop_after_n_frames", -1);

    int gray_threshold = 80;
    int max_blob_size = 24;
    double alpha, beta;
    calcAlphaBeta(gray_threshold, 50, 600, alpha, beta);

    convert_to_gray_pattern->setParameter("alpha", alpha);
    convert_to_gray_pattern->setParameter("beta", beta);

    undistort_pattern->setParameter("optimizeIntrinsics", true);
    undistort_pattern->setParameter("alpha", 0);

    blob_detection_pattern->setParameter("FilterByArea", true);
    blob_detection_pattern->setParameter("FilterByCircularity", true);
    blob_detection_pattern->setParameter("FilterByInertia", true);

    blob_detection_pattern->setParameter("MaxArea", max_blob_size * max_blob_size);
    blob_detection_pattern->setParameter("MinArea", 3 * 3);
    blob_detection_pattern->setParameter("MaxCircularity", 1.0);
    blob_detection_pattern->setParameter("MinCircularity", 0.70); //0.785
    blob_detection_pattern->setParameter("MaxInertiaRatio", 1.0);
    blob_detection_pattern->setParameter("MinInertiaRatio", 0.5);
    blob_detection_pattern->setParameter("MaxThreshold", 255);
    blob_detection_pattern->setParameter("MinThreshold", gray_threshold);
    blob_detection_pattern->setParameter("MinDistBetweenBlobs", 2.0);

//    refine_circles_pattern->setParameter("Threshold", gray_threshold);
//    refine_circles_pattern->setParameter("MaxRadius", max_blob_size/2);

    // setup connections
    graph->connect(get_name("source"), "output_ir", get_name("convert_to_gray"), "input");

    graph->connect(get_name("convert_to_gray"), "output", get_name("undistort"), "input");
    graph->connect(get_name("source"), "output_ir_calibration", get_name("undistort"), "input_calibration");
    graph->connect(get_name("undistort"), "output", get_name("blob_detection"), "input");
    //graph->connect(getName("undistort"), "output", getName("refine_circles"), "input");
    //graph->connect(getName("blob_detection"), "output", getName("refine_circles"), "input_points");
}

int main(int argc, char **argv) {

    using namespace traact;
    using namespace traact::dataflow;
    using namespace traact::facade;

    signal(SIGINT, ctrlC);

    util::initLogging(spdlog::level::debug);

    DefaultInstanceGraphPtr graph = std::make_shared<DefaultInstanceGraph>("tracking_from_mkv");

    int camera_count = 5;
    std::string video_pattern = "/home/frieder/data/recording_20210611_calib1/cn{0:02d}/k4a_capture.mkv";
    //std::string init_file_pattern = "/home/frieder/data/recording_20210611_calib1/cn{0:02d}/new_camera2world.json";
    std::string init_file_pattern = "/home/frieder/data/recording_20210611_calib1/calibration/cn{0:02d}/camera2world_opencv.json";

    std::string result_file_pattern =           "/home/frieder/data/recording_20210611_calib1/data2/{0}.json";
    std::string result_camera_file_pattern =    "/home/frieder/data/recording_20210611_calib1/data2/{0}_{1:02d}.json";

    // prepare a RawApplicationSyncSink with all ports used for debug rendering
    auto debug_pattern = my_facade.instantiatePattern("RawApplicationSyncSink");
    for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
        debug_pattern->addConsumerPort(getRawAppPortName(kDebugImage, camera_index), traact::vision::ImageHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kDebugPoint2D, camera_index), traact::vision::KeyPointListHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kDebugCalibration, camera_index), traact::vision::CameraCalibrationHeader::NativeTypeName);

        debug_pattern->addConsumerPort(getRawAppPortName(kDebugPoint3D, camera_index), traact::vision::Position3DListHeader::NativeTypeName);
        debug_pattern->addConsumerPort(getRawAppPortName(kDebugPose6D, camera_index), traact::spatial::Pose6DHeader::NativeTypeName);
    }

    graph->addPattern("debug_sink", debug_pattern);

    for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
        auto video_file =
            fmt::format(video_pattern, camera_index + 1);
        addBasicProcessing(graph, camera_index, video_file);
        addDebugView(graph, camera_index);

    }
    //addInitTracking(graph, camera_count, init_file_pattern);
    addPointEstimation(graph, camera_count, init_file_pattern);
    addOutsideInTargetTracking(graph, camera_count);
    //addOutsideInTargetRecord(graph, camera_count, result_file_pattern, result_camera_file_pattern);

    buffer::TimeDomainManagerConfig td_config;
    td_config.time_domain = 0;
    td_config.ringbuffer_size = 3;

    td_config.source_mode = SourceMode::WAIT_FOR_BUFFER;
    td_config.missing_source_event_mode = MissingSourceEventMode::WAIT_FOR_EVENT;
    td_config.max_offset = std::chrono::milliseconds(15);
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

//    my_facade.loadDataflow(graph);
//
//    my_facade.blockingStart();

    SPDLOG_INFO("exit program");

    //SPDLOG_INFO("run the same dataflow again using: traactConsole {0}", filename);

    return 0;
}

