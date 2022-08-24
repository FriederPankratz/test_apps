#include <iostream>

#include <traact/traact.h>

#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <traact/serialization/JsonGraphInstance.h>

#include <traact/util/Logging.h>
#include <signal.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "util/YAMLUtil.h"
#include <traact/vision.h>
#include <traact/point_cloud.h>

using namespace traact;
using namespace traact::dataflow;

bool running = true;
traact::facade::DefaultFacade my_facade;

void ctrlC(int i) {
    SPDLOG_INFO("User requested exit (Ctrl-C).");
    my_facade.stop();
}

const static std::string kSingleWindow{"window"};
const static std::string kSceneWindow{"scene"};

const static std::string kImage{"image"};
const static std::string kPointCloudVertex{"pointCloud_vertex"};
const static std::string kPointCloudColor{"pointCloud_color"};
const static std::string kPointCloudPose{"pose_origin"};

const static std::string kMarkerPose{"pose_marker_{0}_pose"};

const static std::string kDebugCalibration{"calibration"};
const static std::string kDebugPose{"pose_{0}"};

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

std::string getIdxName(std::string name, int idx) {
    return fmt::format("{0}_{1}", name, idx);
}

class TraactConfig {
 private:
    std::vector<int> camera_ids_;
    std::map<int, int> camera_id_to_index_;
    std::map<int, int> camera_index_to_id_;

    std::map<int, int> marker_index_to_id_;
    std::map<int, int> marker_id_to_index_;

    int main_camera_id_;
    int origin_marker_id_;
    size_t camera_count_;
    std::string origin_to_camera_file_pattern_;
    std::string origin_to_marker_file;
    DefaultInstanceGraphPtr graph_;
    bool use_shm_;
    pattern::instance::PatternInstance::Ptr register_icp_pattern_;
    bool add_single_windows_;
 public:

    traact::DefaultInstanceGraphPtr create_config(const YAML::Node &config) {
        graph_ = std::make_shared<DefaultInstanceGraph>(config["name"].as<std::string>());
        main_camera_id_ = config["main_camera"].as<int>();
        origin_marker_id_ = config["origin_marker_id"].as<int>();
        use_shm_ = config["use_shm"].as<bool>();
        add_single_windows_ = config["add_single_windows"].as<bool>();

        origin_to_camera_file_pattern_ = config["origin_to_camera_file_pattern"].as<std::string>();
        origin_to_marker_file = config["origin_to_marker_file"].as<std::string>();
        for (const auto &camera_node : config["cameras"]) {
            camera_ids_.push_back(camera_node.as<int>());
            camera_id_to_index_.emplace(camera_node.as<int>(), camera_ids_.size() - 1);
            camera_index_to_id_.emplace(camera_ids_.size() - 1, camera_node.as<int>());
        }
        camera_count_ = camera_ids_.size();

        graph_->timedomain_configs[0] = configureTimeDomain(config["dataflow"]);

        auto &marker_tracker = config["marker_tracker"];
        configureMarker(marker_tracker);

        configureRawApplicationSink();

        addUniquePatterns();




        for (int camera_index = 0; camera_index < camera_count_; ++camera_index) {
            addProcessing(camera_index);

            addMarkerTracker(marker_tracker, camera_index);

            connectTracker(camera_index);
            connectDebugOutput(camera_index);
        }

        addCameraPatterns(config);

        return graph_;
    }

    buffer::TimeDomainManagerConfig configureTimeDomain(const YAML::Node &config) const {
        buffer::TimeDomainManagerConfig td_config;
        td_config.time_domain = 0;
        td_config.ringbuffer_size = config["buffer_size"].as<size_t>();
        td_config.max_offset = std::chrono::milliseconds(config["max_offset"].as<int>());
        td_config.max_delay = std::chrono::milliseconds(100);
        td_config.sensor_frequency = config["sensor_frequency"].as<double>();
        td_config.cpu_count = config["cpu_count"].as<int>();

        if (use_shm_) {
            td_config.source_mode = SourceMode::IMMEDIATE_RETURN;
            td_config.missing_source_event_mode = MissingSourceEventMode::WAIT_FOR_EVENT;
        } else {
            td_config.source_mode = SourceMode::WAIT_FOR_BUFFER;
            td_config.missing_source_event_mode = MissingSourceEventMode::WAIT_FOR_EVENT;
        }
        return td_config;
    }

    void configureMarker(const YAML::Node &config) {
        auto pattern = config["pattern"].as<std::string>();

        if (pattern == "ApriltagTracker" || pattern == "OpenCvArucoTracker") {
            int marker_index{0};
            for (const auto &marker_node : config["marker"]) {
                auto parameter = marker_node.second;
                auto marker_id = parameter["marker_id"].as<int>();
                marker_index_to_id_.emplace(marker_index, marker_id);
                marker_id_to_index_.emplace(marker_id, marker_index);
                marker_index++;
            }
        } else if (pattern == "ArucoFractalTracker") {
            marker_index_to_id_.emplace(0,0);
            marker_id_to_index_.emplace(0,0);
        } else {
            SPDLOG_ERROR("invalid marker tracker type {0}", pattern);
        }
    }

    void configureRawApplicationSink() {
        // prepare a RawApplicationSyncSink with all ports used for debug rendering
        auto debug_pattern = my_facade.instantiatePattern("RawApplicationSyncSink");
        for (int camera_index = 0; camera_index < camera_count_; ++camera_index) {
            if (add_single_windows_) {
                debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow, camera_index, kImage),
                                               traact::vision::ImageHeader::NativeTypeName);
                debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow, camera_index, kDebugCalibration),
                                               traact::vision::CameraCalibrationHeader::NativeTypeName);
            }

            debug_pattern->addConsumerPort(getRawAppPortName(kSceneWindow, camera_index, kPointCloudVertex),
                                           traact::vision::GpuImageHeader::NativeTypeName);
            debug_pattern->addConsumerPort(getRawAppPortName(kSceneWindow, camera_index, kPointCloudColor),
                                           traact::vision::GpuImageHeader::NativeTypeName);

            //debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow, std::to_string(camera_index), "origin"), traact::spatial::Pose6DHeader::NativeTypeName);

            debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow,
                                                              fmt::format("color{0}", camera_index),
                                                              std::to_string(camera_index)),
                                           traact::spatial::Pose6DHeader::NativeTypeName);

            for (int marker_index = 0; marker_index < marker_index_to_id_.size(); ++marker_index) {
                auto marker_id = marker_index_to_id_[marker_index];

                debug_pattern->addConsumerPort(getPoseAppPortName(kSceneWindow,
                                                                  fmt::format("cam{0}marker{1}", camera_index, marker_id),
                                                                  fmt::format("color{0}", camera_index)),
                                               traact::spatial::Pose6DHeader::NativeTypeName);
                if (add_single_windows_) {
                    debug_pattern->addConsumerPort(getRawAppPortName(kSingleWindow,
                                                                     camera_index,
                                                                     fmt::format(kDebugPose, marker_id)),
                                                   traact::spatial::Pose6DHeader::NativeTypeName);
                }

            }

        }

        graph_->addPattern("debug_sink", debug_pattern);
    }

    void addUniquePatterns() {
        auto origin_to_marker_pattern =
            graph_->addPattern(getCalibReaderComponentName("origin", "marker"),
                               my_facade.instantiatePattern("FileReaderWriterRead_cereal_traact::spatial::Pose6D"));
        auto origin_to_marker_write_pattern =
            graph_->addPattern(getCalibWriterComponentName("origin", "marker"),
                               my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));
        origin_to_marker_pattern->setParameter("file", origin_to_marker_file);
        origin_to_marker_pattern->setParameter("CoordinateSystem", "OpenGL");
        origin_to_marker_write_pattern->setParameter("file", origin_to_marker_file);
        origin_to_marker_write_pattern->setParameter("CoordinateSystem", "OpenGL");
        graph_->connect(getCalibReaderComponentName("origin", "marker"),
                        "output",
                        getCalibWriterComponentName("origin", "marker"),
                        "input");

        auto register_using_marker_pattern =
            graph_->addPattern("register_using_marker", my_facade.instantiatePattern("SyncUserEvent"));
        auto register_using_icp_pattern =
            graph_->addPattern("register_using_icp", my_facade.instantiatePattern("SyncUserEvent"));

//        register_icp_pattern_ =
//            graph_->addPattern("register_icp", my_facade.instantiatePattern("Open3DMultiCameraColorICP"));
        register_icp_pattern_ =
            graph_->addPattern("register_icp", my_facade.instantiatePattern("Open3DMultiwayRegistration"));
        register_icp_pattern_->setParameter("reference_node", camera_id_to_index_[main_camera_id_]);
    }

    void addCameraPatterns(const YAML::Node &config) {
        if (use_shm_) {
            addShmSource(config["shm_stream"]);
        } else {
            addVideoSource(config["video_files"]);
        }
    }

    void addShmSource(const YAML::Node config) {
        auto image_stream = config["image_stream"].as<std::string>();
        auto source_pattern =
            graph_->addPattern("image_source", my_facade.instantiatePattern("artekmed::ShmCompositeBufferSource"));
        source_pattern->setParameter("stream", image_stream);
        for (int camera_index = 0; camera_index < camera_count_; ++camera_index) {
            addShmSource(camera_index, source_pattern);
        }
    }
    void addShmSource(int camera_index, traact::DefaultPatternInstancePtr& source_pattern) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        auto camera_id = camera_index_to_id_[camera_index];



        auto &depth_image = source_pattern->instantiatePortGroup("depth_image");
        depth_image.setParameter("channel", fmt::format("camera{0:02d}_depthimage", camera_id));
        auto &color_image = source_pattern->instantiatePortGroup("color_image");
        color_image.setParameter("channel", fmt::format("camera{0:02d}_colorimage", camera_id));
        auto calibration_pattern =
            graph_->addPattern(get_name("calibration"),
                              my_facade.instantiatePattern("artekmed::ShmSensorCalibrationSource"));
        calibration_pattern->setParameter("stream", fmt::format("camera{0:02d}", camera_id));

        graph_->connect(source_pattern->getName(), depth_image.getProducerPortName("output"),
                        get_name("upload_depth"), "input");

        graph_->connect(source_pattern->getName(), color_image.getProducerPortName("output"),
                        get_name("upload_color"), "input");

        graph_->connect(get_name("calibration"), "output_xy_table",
                        get_name("upload_xy_table"), "input");

        graph_->connect(get_name("calibration"), "output_color_calibration",
                        get_name("undistort_color"), "input_calibration");

        graph_->connect(get_name("calibration"), "output_color_to_ir",
                        get_name("depth_to_color"), "input");

        graph_->connect(get_name("calibration"), "output_color_to_ir",
                        get_name("origin_to_depth_camera_mul"), "input_b");

        graph_->connect(get_name("calibration"), "output_color_calibration",
                        get_name("color_point_cloud"),
                        "input_color_calibration");
        graph_->connect(get_name("calibration"), "output_color_to_ir",
                        get_name("color_point_cloud"),
                        "input_color_to_depth");


    }
    void addVideoSource(const YAML::Node config) {
        auto video_file_pattern = config["video_file_pattern"].as<std::string>();
        for (int camera_index = 0; camera_index < camera_count_; ++camera_index) {
            addVideoSource(camera_index, video_file_pattern);
        }
    }

    void addVideoSource(int camera_index, const std::string &video_file_pattern) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        auto source_pattern =
            graph_->addPattern(getIdxName("source", camera_index),
                               my_facade.instantiatePattern("traact::component::kinect::KinectAzureSingleFilePlayer"));
        source_pattern->setParameter("file", fmt::format(video_file_pattern, camera_index_to_id_[camera_index]));
        source_pattern->setParameter("stop_after_n_frames", -1);
        source_pattern->setParameter("send_same_frame_as_new_after_stop", true);


        graph_->connect(get_name("source"), "output_depth",
                        get_name("upload_depth"), "input");

        graph_->connect(get_name("source"), "output_xy_table",
                        get_name("upload_xy_table"), "input");

        graph_->connect(get_name("source"), "output",
                        get_name("upload_color"), "input");

        graph_->connect(get_name("source"), "output_calibration",
                        get_name("undistort_color"), "input_calibration");

        graph_->connect(get_name("source"), "output_color_to_depth",
                        get_name("depth_to_color"), "input");

        graph_->connect(get_name("source"), "output_color_to_depth",
                        get_name("origin_to_depth_camera_mul"), "input_b");

        graph_->connect(get_name("source"),
                        "output_calibration",
                        get_name("color_point_cloud"),
                        "input_color_calibration");
        graph_->connect(get_name("source"),
                        "output_color_to_depth",
                        get_name("color_point_cloud"),
                        "input_color_to_depth");
    }

    void addMarkerTracker(const YAML::Node &config, int camera_index) {
        auto pattern = config["pattern"].as<std::string>();
        if (pattern == "ApriltagTracker") {
            addAprilTagTracker(config, camera_index);
        } else if (pattern == "OpenCvArucoTracker") {
            addArucoTracker(config, camera_index);
        } else if (pattern == "ArucoFractalTracker") {
            addArucoFractalTracker(config, camera_index);
        } else {
            SPDLOG_ERROR("invalid marker tracker type {0}", pattern);
        }
    }

    void addArucoTracker(const YAML::Node &config, int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };
        auto origin_tracker_pattern =
            graph_->addPattern(get_name("origin_tracker"), my_facade.instantiatePattern("OpenCvArucoTracker"));
        origin_tracker_pattern->setParameter("dictionary", config["dictionary"].as<std::string>());
        addTacker(config, camera_index, origin_tracker_pattern);
    }

    void addAprilTagTracker(const YAML::Node &config, int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };
        auto origin_tracker_pattern =
            graph_->addPattern(get_name("origin_tracker"), my_facade.instantiatePattern("ApriltagTracker"));
        addTacker(config, camera_index, origin_tracker_pattern);

    }

    void addArucoFractalTracker(const YAML::Node &config, int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };
        auto origin_tracker_pattern =
            graph_->addPattern(get_name("origin_tracker"), my_facade.instantiatePattern("ArucoFractalTracker"));
        origin_tracker_pattern->setParameter("marker_config", config["marker_config"].as<std::string>());
        origin_tracker_pattern->setParameter("marker_size", config["marker_size"].as<double>());
        addTacker(config, camera_index, origin_tracker_pattern);

    }

    void addTacker(const YAML::Node &config,
                   int camera_index,
                   traact::DefaultPatternInstancePtr &origin_tracker_pattern) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        auto pattern = config["pattern"].as<std::string>();


        for (const auto &marker_node : config["marker"]) {
            auto parameter = marker_node.second;

            auto &marker = origin_tracker_pattern->instantiatePortGroup("output_pose");
            auto marker_id = parameter["marker_id"].as<int>();


            marker.setParameter("marker_id", marker_id);
            marker.setParameter("marker_size", parameter["marker_size"].as<double>());

            graph_->connect(get_name("origin_tracker"),
                            marker.getProducerPortName("output"),
                            "debug_sink",
                            getPoseAppPortName(kSceneWindow,
                                               fmt::format("cam{0}marker{1}", camera_index, marker_id),
                                               fmt::format("color{0}", camera_index)));

            if (add_single_windows_) {
                graph_->connect(get_name("origin_tracker"),
                                marker.getProducerPortName("output"),
                                "debug_sink",
                                getRawAppPortName(kSingleWindow, camera_index, fmt::format(kDebugPose, marker_id)));
            }

            if (marker_id == origin_marker_id_) {
                graph_->connect(get_name("origin_tracker"),
                                marker.getProducerPortName("output"),
                                get_name("marker_to_camera"),
                                "input");
            }
        }
    }

    void addProcessing(int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        auto depth_to_color_pattern =
            graph_->addPattern(get_name("depth_to_color"), my_facade.instantiatePattern("InversionPose6D"));

        auto origin_to_camera_pattern =
            graph_->addPattern(getCalibReaderComponentName("origin", std::to_string(camera_index)),
                               my_facade.instantiatePattern("FileReaderWriterRead_cereal_traact::spatial::Pose6D"));
        auto origin_to_camera_write_pattern =
            graph_->addPattern(getCalibWriterComponentName("origin", std::to_string(camera_index)),
                               my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));

        auto origin_to_camera_color_mul_pattern =
            graph_->addPattern(get_name("origin_to_color_camera_mul"),
                               my_facade.instantiatePattern("MultiplicationPose6DPose6D"));
        auto origin_to_camera_depth_mul_pattern =
            graph_->addPattern(get_name("origin_to_depth_camera_mul"),
                               my_facade.instantiatePattern("MultiplicationPose6DPose6D"));

        auto marker_to_camera_pattern =
            graph_->addPattern(get_name("marker_to_camera"), my_facade.instantiatePattern("InversionPose6D"));

        auto gate_origin_to_camera_pattern =
            graph_->addPattern(get_name("gate_origin_to_camera"),
                               my_facade.instantiatePattern("Gate_traact::spatial::Pose6D"));

        auto gate_point_cloud_pattern =
            graph_->addPattern(get_name("gate_point_cloud"),
                               my_facade.instantiatePattern("Gate_traact::vision::GpuImage"));
        auto gate_point_color_pattern =
            graph_->addPattern(get_name("gate_point_color"),
                               my_facade.instantiatePattern("Gate_traact::vision::GpuImage"));

        auto download_point_image_pattern =
            graph_->addPattern(get_name("download_point_cloud"), my_facade.instantiatePattern("OpenCvCudaDownload"));
        auto download_point_color_pattern =
            graph_->addPattern(get_name("download_point_color"), my_facade.instantiatePattern("OpenCvCudaDownload"));
        auto build_point_cloud_pattern =
            graph_->addPattern(get_name("build_point_cloud"), my_facade.instantiatePattern("Open3DBuildPointCloud"));

        auto world_to_camera_file = fmt::format(origin_to_camera_file_pattern_, camera_index_to_id_[camera_index]);
        origin_to_camera_pattern->setParameter("file", world_to_camera_file);
        origin_to_camera_pattern->setParameter("CoordinateSystem", "OpenGL");
        origin_to_camera_write_pattern->setParameter("file", world_to_camera_file);
        origin_to_camera_write_pattern->setParameter("CoordinateSystem", "OpenGL");

        download_point_image_pattern->setParameter("cuda_graph", "download_for_icp");
        download_point_color_pattern->setParameter("cuda_graph", "download_for_icp");



        // create point cloud
        auto upload_depth_pattern =
            graph_->addPattern(get_name("upload_depth"), my_facade.instantiatePattern("OpenCvCudaUpload"));
        auto upload_xy_table_pattern =
            graph_->addPattern(get_name("upload_xy_table"), my_facade.instantiatePattern("OpenCvCudaUpload"));

        auto create_point_cloud_pattern =
            graph_->addPattern(get_name("create_point_cloud"), my_facade.instantiatePattern("CudaCreatePointCloud"));

        graph_->connect(get_name("upload_depth"), "output", get_name("create_point_cloud"), "input");
        graph_->connect(get_name("upload_xy_table"), "output", get_name("create_point_cloud"), "input_xy_table");
        // color point cloud
        auto color_point_cloud_pattern =
            graph_->addPattern(get_name("color_point_cloud"), my_facade.instantiatePattern("CudaColorPointCloud"));
        auto upload_color_pattern =
            graph_->addPattern(get_name("upload_color"), my_facade.instantiatePattern("OpenCvCudaUpload"));
        graph_->connect(get_name("create_point_cloud"), "output", get_name("color_point_cloud"), "input");
        graph_->connect(get_name("upload_color"), "output", get_name("color_point_cloud"), "input_color");


        // create undistorted gray image for tracker
        auto undistort_color_pattern =
            graph_->addPattern(get_name("undistort_color"), my_facade.instantiatePattern("OpenCvCudaUndistortImage"));
        auto color_to_gray_pattern =
            graph_->addPattern(get_name("color_to_gray"), my_facade.instantiatePattern("OpenCvCudaCvtColor"));
        auto download_gray_pattern =
            graph_->addPattern(get_name("download_gray"), my_facade.instantiatePattern("OpenCvCudaDownload"));

        graph_->connect(get_name("upload_color"), "output", get_name("undistort_color"), "input");
        // connect calibration in connectSources()
        graph_->connect(get_name("undistort_color"), "output", get_name("color_to_gray"), "input");
        graph_->connect(get_name("color_to_gray"), "output", get_name("download_gray"), "input");

        if(add_single_windows_){
            auto download_color_pattern =
                graph_->addPattern(get_name("download_color"), my_facade.instantiatePattern("OpenCvCudaDownload"));
            graph_->connect(get_name("undistort_color"), "output", get_name("download_color"), "input");
        }




        // control flow for marker and icp registration
        graph_->connect(getCalibReaderComponentName("origin", "marker"), "output",
                        get_name("origin_to_color_camera_mul"), "input_a");

        graph_->connect(get_name("marker_to_camera"), "output",
                        get_name("origin_to_color_camera_mul"), "input_b");

        graph_->connect(get_name("origin_to_color_camera_mul"), "output",
                        get_name("origin_to_depth_camera_mul"), "input_a");

        graph_->connect(get_name("origin_to_depth_camera_mul"), "output",
                        get_name("gate_origin_to_camera"), "input");

        graph_->connect("register_using_marker", "output",
                        get_name("gate_origin_to_camera"), "input_event");

        graph_->connect(get_name("gate_origin_to_camera"), "output",
                        getCalibWriterComponentName("origin", std::to_string(camera_index)), "input");

        graph_->connect(get_name("create_point_cloud"), "output", get_name("gate_point_cloud"), "input");
        graph_->connect("register_using_icp", "output", get_name("gate_point_cloud"), "input_event");

        graph_->connect(get_name("color_point_cloud"), "output", get_name("gate_point_color"), "input");
        graph_->connect("register_using_icp", "output", get_name("gate_point_color"), "input_event");

        graph_->connect(get_name("gate_point_cloud"), "output", get_name("download_point_cloud"), "input");
        graph_->connect(get_name("gate_point_color"), "output", get_name("download_point_color"), "input");

        graph_->connect(get_name("download_point_cloud"), "output", get_name("build_point_cloud"), "input");
        graph_->connect(get_name("download_point_color"), "output", get_name("build_point_cloud"), "input_color");

        auto icp_camera = register_icp_pattern_->instantiatePortGroup("camera");
        auto origin_to_camera_write_icp_pattern =
            graph_->addPattern(get_name("origin_to_camera_write_icp"),
                               my_facade.instantiatePattern("FileReaderWriterWrite_cereal_traact::spatial::Pose6D"));
        origin_to_camera_write_icp_pattern->setParameter("file", world_to_camera_file);
        origin_to_camera_write_icp_pattern->setParameter("CoordinateSystem", "OpenGL");

        graph_->connect(getCalibReaderComponentName("origin", std::to_string(camera_index)), "output",
                        "register_icp", icp_camera.getConsumerPortName("input_pose"));

        graph_->connect(get_name("build_point_cloud"), "output",
                        "register_icp", icp_camera.getConsumerPortName("input_cloud"));

        graph_->connect("register_icp", icp_camera.getProducerPortName("output"),
                        get_name("origin_to_camera_write_icp"), "input");
    }

    void connectTracker(int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        // marker tracker has always the same input
        graph_->connect(get_name("download_gray"), "output", get_name("origin_tracker"), "input");
        graph_->connect(get_name("undistort_color"),
                        "output_calibration",
                        get_name("origin_tracker"),
                        "input_calibration");

    }

    void connectDebugOutput(int camera_index) {
        auto get_name = [camera_index](const std::string &name) {
            return getIdxName(name, camera_index);
        };

        if (add_single_windows_) {
            graph_->connect(get_name("download_color"), "output",
                            "debug_sink", getRawAppPortName(kSingleWindow, camera_index, kImage));

            graph_->connect(get_name("undistort_color"), "output_calibration",
                            "debug_sink", getRawAppPortName(kSingleWindow, camera_index, kDebugCalibration));
        }

        graph_->connect(get_name("create_point_cloud"), "output",
                        "debug_sink", getRawAppPortName(kSceneWindow, camera_index, kPointCloudVertex));

        graph_->connect(get_name("color_point_cloud"), "output",
                        "debug_sink", getRawAppPortName(kSceneWindow, camera_index, kPointCloudColor));

        graph_->connect(get_name("depth_to_color"), "output",
                        "debug_sink", getPoseAppPortName(kSceneWindow,
                                                         fmt::format("color{0}", camera_index),
                                                         std::to_string(camera_index)));


        //graph_->connect(getCalibComponentName("origin", std::to_string(index)), "output", "debug_sink", getCalibAppPortName(kSceneWindow, std::to_string(camera_index), "origin"));

    }

};

bool hasConfigError(const YAML::Node &config) {
    if (!util::HasValue("name", config)) {
        return true;
    }
    if (!util::HasValue("main_camera", config)) {
        return true;
    }
    if (!util::HasValue("origin_marker_id", config)) {
        return true;
    }
    if (!util::HasValue("dataflow", config)) {
        return true;
    }
    if (!util::HasValue("cameras", config)) {
        return true;
    }
    if (!util::HasValue("marker_tracker", config)) {
        return true;
    }
    if (!util::HasValue("origin_to_camera_file_pattern", config)) {
        return true;
    }
    if (!util::HasValue("origin_to_marker_file", config)) {
        return true;
    }

    return false;
}

void writeTraactFile(const DefaultInstanceGraphPtr &graph) {

    auto not_runnable = graph->checkRunnable();
    if (not_runnable) {
        SPDLOG_ERROR(not_runnable.value());
        return;
    }
    std::string filename = graph->name + ".json";
    nlohmann::json jsongraph;
    ns::to_json(jsongraph, *graph);

    std::ofstream graph_file;
    graph_file.open(filename);
    graph_file << jsongraph.dump(4);
    graph_file.close();
}

int main(int argc, char **argv) {

    using namespace traact;
    using namespace traact::dataflow;
    using namespace traact::facade;

    signal(SIGINT, ctrlC);

    util::initLogging(spdlog::level::trace);

    const std::string keys =
        "{help h usage ? |      | print this message   }"
        "{@config        |<none>| config file          }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("traact shm dataflow generator v1.0.0");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }
    if (!parser.has("@config")) {
        SPDLOG_ERROR("missing config file");
        parser.printMessage();
        return 0;
    }

    std::string config_file = parser.get<std::string>(0);
    auto all_config = YAML::LoadFile(config_file);
    if (!util::HasValue("config", all_config)) {
        return 0;
    }
    auto config = all_config["config"];

    if (hasConfigError(config)) {
        return 0;
    }

    auto base_name = config["name"].as<std::string>();

    if (config["video_files"]) {
        config["name"] = fmt::format("{0}_{1}", base_name, "mkv");
        config["use_shm"] = false;
        TraactConfig config_generator;
        auto graph = config_generator.create_config(config);
        writeTraactFile(graph);
    } else {
        SPDLOG_WARN("not generating dataflow for video files, no \"video_files\" section defined");
    }

    if (config["shm_stream"]) {
        config["name"] = fmt::format("{0}_{1}", base_name, "shm");
        config["use_shm"] = true;
        TraactConfig config_generator;
        auto graph = config_generator.create_config(config);
        writeTraactFile(graph);
    } else {
        SPDLOG_WARN("not generating dataflow for shm files, no \"shm_stream\" section defined");
    }
}
