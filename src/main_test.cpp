#include <iostream>

#include <traact/traact.h>

#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <traact/serialization/JsonGraphInstance.h>

#include "util/YAMLUtil.h"
#include <signal.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <traact/point_cloud.h>
#include <traact/util/Logging.h>
#include <traact/vision.h>

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


    configureRawApplicationSink();



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
                    "debug_sink", getPoseAppPortName(kSceneWindow, fmt::format("color{0}", camera_index), std::to_string(camera_index)));

    //graph_->connect(getCalibComponentName("origin", std::to_string(index)), "output", "debug_sink", getCalibAppPortName(kSceneWindow, std::to_string(camera_index), "origin"));
  }
};


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

//  if (config["video_files"]) {
//    config["name"] = fmt::format("{0}_{1}", base_name, "mkv");
//    config["use_shm"] = false;
//    TraactConfig config_generator;
//    auto graph = config_generator.create_config(config);
//    writeTraactFile(graph);
//  } else {
//    SPDLOG_WARN("not generating dataflow for video files, no \"video_files\" section defined");
//  }
//
//  if (config["shm_stream"]) {
//    config["name"] = fmt::format("{0}_{1}", base_name, "shm");
//    config["use_shm"] = true;
//    TraactConfig config_generator;
//    auto graph = config_generator.create_config(config);
//    writeTraactFile(graph);
//  } else {
//    SPDLOG_WARN("not generating dataflow for shm files, no \"shm_stream\" section defined");
//  }
}
