// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "mvs/patch_match.h"

#include <numeric>
#include <unordered_set>
#include <random>

#include "mvs/consistency_graph.h"
#include "mvs/patch_match_cuda.h"
#include "mvs/workspace.h"
#include "util/math.h"
#include "util/misc.h"

#define PrintOption(option) std::cout << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

PatchMatch::PatchMatch(const PatchMatchOptions& options, const Problem& problem)
    : options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() {}

void PatchMatchOptions::Print() const {
  PrintHeading2("PatchMatchOptions");
  PrintOption(max_image_size);
  PrintOption(gpu_index);
  PrintOption(depth_min);
  PrintOption(depth_max);
  PrintOption(window_radius);
  PrintOption(window_step);
  PrintOption(sigma_spatial);
  PrintOption(sigma_color);
  PrintOption(num_samples);
  PrintOption(ncc_sigma);
  PrintOption(min_triangulation_angle);
  PrintOption(incident_angle_sigma);
  PrintOption(num_iterations);
  PrintOption(geom_consistency);
  PrintOption(geom_consistency_regularizer);
  PrintOption(geom_consistency_max_cost);
  PrintOption(filter);
  PrintOption(filter_min_ncc);
  PrintOption(filter_min_triangulation_angle);
  PrintOption(filter_min_num_consistent);
  PrintOption(filter_geom_consistency_max_cost);
  PrintOption(write_consistency_graph);
  PrintOption(allow_missing_files);
  PrintOption(pos_min_dis);
  PrintOption(pos_max_dis);
  PrintOption(ort_min_dis);
  PrintOption(ort_max_dis);
}

void PatchMatch::Problem::Print() const {
  PrintHeading2("PatchMatch::Problem");

  PrintOption(ref_image_idx);

  std::cout << "src_image_idxs: ";
  if (!src_image_idxs.empty()) {
    for (size_t i = 0; i < src_image_idxs.size() - 1; ++i) {
      std::cout << src_image_idxs[i] << " ";
    }
    std::cout << src_image_idxs.back() << std::endl;
  } else {
    std::cout << std::endl;
  }
}

void PatchMatch::Check() const {
  CHECK(options_.Check());

  CHECK(!options_.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1);
  CHECK_GE(gpu_indices[0], -1);

  CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency) {
    CHECK_NOTNULL(problem_.depth_maps);
    CHECK_NOTNULL(problem_.normal_maps);
    CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }

  CHECK_GT(problem_.src_image_idxs.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
                                  problem_.src_image_idxs.end());
  unique_image_idxs.insert(problem_.ref_image_idx);
  CHECK_EQ(problem_.src_image_idxs.size() + 1, unique_image_idxs.size());

  // Check that input data is well-formed.
  for (const int image_idx : unique_image_idxs) {
    CHECK_GE(image_idx, 0) << image_idx;
    CHECK_LT(image_idx, problem_.images->size()) << image_idx;

    const Image& image = problem_.images->at(image_idx);
    CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
    CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
    CHECK(image.GetBitmap().IsGrey()) << image_idx;
    CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
    CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options_.geom_consistency) {
      CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
      CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

void PatchMatch::Run() {
  PrintHeading2("PatchMatch::Run");

  Check();

  patch_match_cuda_.reset(new PatchMatchCuda(options_, problem_));
  patch_match_cuda_->Run();
}

DepthMap PatchMatch::GetDepthMap() const {
  return patch_match_cuda_->GetDepthMap();
}

NormalMap PatchMatch::GetNormalMap() const {
  return patch_match_cuda_->GetNormalMap();
}

Mat<float> PatchMatch::GetSelProbMap() const {
  return patch_match_cuda_->GetSelProbMap();
}

ConsistencyGraph PatchMatch::GetConsistencyGraph() const {
  const auto& ref_image = problem_.images->at(problem_.ref_image_idx);
  return ConsistencyGraph(ref_image.GetWidth(), ref_image.GetHeight(),
                          patch_match_cuda_->GetConsistentImageIdxs());
}

PatchMatchController::PatchMatchController(const PatchMatchOptions& options,
                                           const std::string& workspace_path,
                                           const std::string& workspace_format,
                                           const std::string& pmvs_option_name,
                                           const std::string& config_path)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      config_path_(config_path) {
  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
}

void PatchMatchController::Run() {
  ReadWorkspace();
  ReadProblems();
  ReadGpuIndices();

  thread_pool_.reset(new ThreadPool(gpu_indices_.size()));

  // If geometric consistency is enabled, then photometric output must be
  // computed first for all images without filtering.
  if (options_.geom_consistency) {
    auto photometric_options = options_;
    photometric_options.geom_consistency = false;
    photometric_options.filter = false;

    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&PatchMatchController::ProcessProblem, this,
                            photometric_options, problem_idx);
    }

    thread_pool_->Wait();
  }

  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(&PatchMatchController::ProcessProblem, this, options_,
                          problem_idx);
  }

  thread_pool_->Wait();

  GetTimer().PrintMinutes();
}

void PatchMatchController::ReadWorkspace() {
  std::cout << "Reading workspace..." << std::endl;

  Workspace::Options workspace_options;

  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = false;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = options_.geom_consistency ? "photometric" : "";

  workspace_.reset(new CachedWorkspace(workspace_options));

  if (workspace_format_lower_case == "pmvs") {
    std::cout << StringPrintf("Importing PMVS workspace (option %s)...",
                              pmvs_option_name_.c_str())
              << std::endl;
    ImportPMVSWorkspace(*workspace_, pmvs_option_name_);
  }

  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatchController::ReadProblems() {
  std::cout << "Reading configuration..." << std::endl;

  problems_.clear();

  const auto& model = workspace_->GetModel();

  const std::string config_path =
      config_path_.empty()
          ? JoinPaths(workspace_path_, workspace_->GetOptions().stereo_folder,
                      "patch-match.cfg")
          : config_path_;
  std::vector<std::string> config = ReadTextFileLines(config_path);

  std::vector<std::map<int, int>> shared_num_points;
  std::vector<std::map<int, float>> triangulation_angles;

  const float min_triangulation_angle_rad =
      DegToRad(options_.min_triangulation_angle);

  std::string ref_image_name;
  std::unordered_set<int> ref_image_idxs;

  struct ProblemConfig {
    std::string ref_image_name;
    std::vector<std::string> src_image_names;
  };
  std::vector<ProblemConfig> problem_configs;

  for (size_t i = 0; i < config.size(); ++i) {
    std::string& config_line = config[i];
    StringTrim(&config_line);

    if (config_line.empty() || config_line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = config_line;
      continue;
    }

    ref_image_idxs.insert(model.GetImageIdx(ref_image_name));

    ProblemConfig problem_config;
    problem_config.ref_image_name = ref_image_name;
    problem_config.src_image_names = CSVToVector<std::string>(config_line);
    problem_configs.push_back(problem_config);

    ref_image_name.clear();
  }

  for (const auto& problem_config : problem_configs) {
    PatchMatch::Problem problem;

    problem.ref_image_idx = model.GetImageIdx(problem_config.ref_image_name);

    if (problem_config.src_image_names.size() == 1 &&
        problem_config.src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_idxs.clear();
      problem.src_image_idxs.reserve(model.images.size() - 1);
      for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        if (static_cast<int>(image_idx) != problem.ref_image_idx) {
          problem.src_image_idxs.push_back(image_idx);
        }
      }
    } 
    else if (problem_config.src_image_names.size() == 2 &&
               problem_config.src_image_names[0] == "__auto__") {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected. Note that images are only
      // selected if some points have a sufficient triangulation angle.

      if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
      }
      if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
      }

      const size_t max_num_src_images =
          std::stoll(problem_config.src_image_names[1]);

      const auto& overlapping_images =
          shared_num_points.at(problem.ref_image_idx);
      const auto& overlapping_triangulation_angles =
          triangulation_angles.at(problem.ref_image_idx);

      std::vector<std::pair<int, int>> src_images;
      src_images.reserve(overlapping_images.size());
      for (const auto& image : overlapping_images) {
        if (overlapping_triangulation_angles.at(image.first) >=
            min_triangulation_angle_rad) {
          src_images.emplace_back(image.first, image.second);
        }
      }

      const size_t eff_max_num_src_images =
          std::min(src_images.size(), max_num_src_images);

      std::partial_sort(src_images.begin(),
                        src_images.begin() + eff_max_num_src_images,
                        src_images.end(),
                        [](const std::pair<int, int>& image1,
                           const std::pair<int, int>& image2) {
                          return image1.second > image2.second;
                        });

      problem.src_image_idxs.reserve(eff_max_num_src_images);
      for (size_t i = 0; i < eff_max_num_src_images; ++i) {
        problem.src_image_idxs.push_back(src_images[i].first);
      }
    }  
    else if(problem_config.src_image_names.size() == 3 &&
               problem_config.src_image_names[0] == "__two-stage__"){
      // perform two-stage select.
      // e.g. __two-stage__ 3 7
      // stage-mode 1: preform 1 stage
      // stage-mode 2: preform 2 stage
      // stage-mode 3: perform 1, 2 stage

      // set the stage-1 threshold
      const float pos_min_dis = (float)options_.pos_min_dis;
      const float pos_max_dis = (float)options_.pos_max_dis;

      const float ort_min_dis = (float)options_.ort_min_dis * M_PI / 180;
      const float ort_max_dis = (float)options_.ort_max_dis * M_PI / 180;

      const int stage_mode = static_cast<int>(std::stoll(problem_config.src_image_names[1]));
      std::cout << "View Selection stage-mode:" << stage_mode << std::endl;
      assert(stage_mode == 1 || stage_mode == 2 || stage_mode == 3);

      const size_t max_num_src_images = std::stoll(problem_config.src_image_names[2]);
      
      // get each view's n and pos
      const auto view_ort = model.ComputeViewRays();
      const auto view_pos = model.ComputeViewPos();

      // candidate_views store the views selected by first stage, (img_id, score) for ref img
      std::vector<std::pair<int, float>> candidate_views;
      Model::Point ref_ort = view_ort.at(problem.ref_image_idx);
      Model::Point ref_pos = view_pos.at(problem.ref_image_idx);

      if(stage_mode == 1 || stage_mode == 3){
        // stage 1 : filter out the view
        for(size_t image_idx = 0; image_idx < model.images.size(); image_idx++){
          if (static_cast<int>(image_idx) != problem.ref_image_idx) {
            Model::Point src_ort = view_ort.at(image_idx);
            Model::Point src_pos = view_pos.at(image_idx);
            float pos_dis = (ref_pos - src_pos).norm();
            float ort_dis = acos( ref_ort.dot(src_ort) / (ref_ort.norm() * src_ort.norm()) );
            if(pos_dis > pos_min_dis && pos_dis < pos_max_dis && 
                ort_dis > ort_min_dis && ort_dis < ort_max_dis){
                  float score = (pos_dis - pos_min_dis) / (pos_max_dis - pos_dis) +
                                  (ort_dis - ort_min_dis) / (ort_max_dis - ort_min_dis);
                  candidate_views.emplace_back(std::make_pair(static_cast<int>(image_idx), score));
            }
          }
        }
      }
      else{
        // use all image as candidate_views and set score to 0.0f if don't use stage 1
        for(size_t image_idx = 0; image_idx < model.images.size(); image_idx++){
          candidate_views.emplace_back(std::make_pair(image_idx, 0.0f));
        }
      }

      std::cout << "candidate views for ref image " << problem.ref_image_idx << " after stage 1:" << std::endl;
      std::cout << "candidate views size: " << candidate_views.size() << "each of them: ";
      for(auto cand: candidate_views){
        std::cout << cand.first << ",";
      }
      std::cout << std::endl;

      // src_images store the final source views, list of (img_id, score)
      std::vector<std::pair<int, float>> src_images;
      if(stage_mode == 2 || stage_mode == 3){
        //perform stage 2 k-means

        // 1. compute features for each view, view_feats store the features of each candidate, (img_id, (geom_dis, point_dis, img_dis))
        std::unordered_map<int, Model::Point> view_feats;
        Bitmap ref_img;
        CHECK(ref_img.Read(workspace_->GetBitmapPath(problem.ref_image_idx), false));
      
        // compute all view center 
        std::vector<Model::Point> proj_centers(model.images.size());
        for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
          const auto& image = model.images[image_idx];
          Model::Point C;
          const float * R = image.GetR();
          const float * T = image.GetR();
          C.x = -(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]);
          C.y = -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]);
          C.z = -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2]);
          proj_centers[image_idx] = C;
        }

        //(img_id, {point_id}), the points in each img_id
        std::unordered_map<int, std::vector<int>> fs;
        //(point_id, score), compute each score for point according to angle between all views contain it
        std::unordered_map<int, float> wn;

        // compute fs and wn
        for (size_t p_id = 0; p_id < model.points.size(); p_id++) {
          auto point = model.points[p_id];
          // compute fs
          for (size_t i = 0; i < point.track.size(); ++i) {
            const int image_idx1 = point.track[i];
            if(image_idx1 == problem.ref_image_idx){
              // this point is in ref_img, then store it
              for(size_t j = 0; j < point.track.size(); ++j){
                if(i != j){
                  fs[point.track[j]].emplace_back(p_id);
                }
              }
              break;
            }
          }
          // compute the wn
          float score = 1.0f;
          for (size_t i = 0; i < point.track.size(); ++i) {
            for(size_t j = 0; j < i; ++j){
              const float angle = model.CalculateTriangulationAnglePoint(
                                     proj_centers.at(i), proj_centers.at(j),
                                     point);
              score *= std::min(( angle * angle / ort_max_dis * ort_max_dis), 1.0f);
            }
          }
          wn[p_id] = score;
        }
        
        const float* ref_K = model.images[problem.ref_image_idx].GetK();
        const float* ref_P = model.images[problem.ref_image_idx].GetP();
        
        // compute features for each view
        for(size_t image_idx = 0; image_idx < model.images.size(); image_idx++){
          if (static_cast<int>(image_idx) != problem.ref_image_idx) {
            // geom dis score
            Model::Point src_ort = view_ort.at(image_idx);
            Model::Point src_pos = view_pos.at(image_idx);
            float pos_dis = (ref_pos - src_pos).norm();
            float ort_dis = acos( ref_ort.dot(src_ort) / (ref_ort.norm() * src_ort.norm()) );
            float geom_dis = 0.5 * (pos_dis - pos_min_dis) / (pos_max_dis - pos_dis) +
                                  0.5 *(ort_dis - ort_min_dis) / (ort_max_dis - ort_min_dis);

            // shared point score
            float point_dis = 0.0f;
            for(auto p_id: fs[image_idx]){
              auto point = model.points[p_id];
              
              const float ref_p_z = ref_P[8] * point.x + ref_P[9] * point.y + ref_P[10] * point.z + ref_P[11];
              const float ref_f = std::min(ref_K[0], ref_K[4]);
              float sr = std::abs(ref_p_z) / ref_f;

              const float* src_K = model.images[image_idx].GetK();
              const float* src_P = model.images[image_idx].GetP();
              const float src_p_z = src_P[8] * point.x + src_P[9] * point.y + src_P[10] * point.z + src_P[11];
              const float src_f = std::min(src_K[0], src_K[4]);
              float sv = std::abs(src_p_z) / src_f;

              const float r = sr / sv;
              float ws = r;
              
              if(r >= 2){
                ws = 2 / r;
              }
              else if(r >= 1 && r < 2){
                ws = 1.0f;
              }

              point_dis += wn[p_id] * ws;
            }

            // img similarity score
            Bitmap src_img;
            CHECK(src_img.Read(workspace_->GetBitmapPath(image_idx), false));
            float img_dis = 1 - ref_img.GetImageSimilarity(src_img);
            //float img_dis = 0.0f;
            view_feats[static_cast<int>(image_idx)] = Model::Point(geom_dis, point_dis, img_dis);
          }
        }
        
        std::cout << "features for each candidate view for ref image " << problem.ref_image_idx << std::endl;
        for(auto cand: candidate_views){
          Model::Point feat = view_feats[cand.first];
          std::cout << "candidate view: " << cand.first << " feat:(" << feat.x << ","  << feat.y << ","  << feat.z << ")" << std::endl;
        }
        std::cout << std::endl;

        // k means
        size_t k = std::min(candidate_views.size(), max_num_src_images);
        std::cout << "src view size for ref image " << problem.ref_image_idx << " is " << k << std::endl;
        float a1 = 1.0f;
        float a2 = 1.0f;
        float a3 = 1.0f;
        
        // center_ids: cand_id, use candidate_views[center_ids[i]].first to get img_id
        std::vector<int> center_ids(candidate_views.size());
        std::iota(center_ids.begin(), center_ids.end(), 0);
        std::random_shuffle (center_ids.begin(), center_ids.end());
        center_ids.reserve(k);

        //store the view belong to each category's. (center_id, {cand_id})
        std::unordered_map<int, std::vector<int>> token;
        
        int iter = 0;
        while(iter < 5){
          std::cout << "Iter" << iter << ":"<< std::endl;
          std::cout << "source views: " << candidate_views.size();
          for(auto cand_id: center_ids){
            std::cout << candidate_views[cand_id].first << ",";
          }
          std::cout << std::endl;

          iter++;

          // change token with new center
          for(size_t i = 0; i < candidate_views.size(); i++){
            int minIndex = -1;
            float minDis = INFINITY;
            Model::Point f = view_feats[candidate_views[i].first];
            // find the min cost center
            for(size_t j = 0; j < k; j++){
              Model::Point center_f = view_feats[candidate_views[center_ids[j]].first];
              float dis = a1 * std::abs(center_f.x - f.x) + a2 * std::abs(center_f.y - f.y) + a3 * std::abs(center_f.z - f.z);
              if(dis < minDis){
                  minDis = dis;
                  minIndex = j;
              }
            }
            //std::cout << "view" << candidate_views[i].first << " belongs to " << minIndex <<std::endl;
            token[minIndex].emplace_back(i);
          }

          // updata new center
          for(size_t j = 0; j < k; j++){
            // compute the avg center each category
            Model::Point new_center_f(0.0f, 0.0f, 0.0f);
            for(auto cand_id: token[j]){
              new_center_f = new_center_f + view_feats[candidate_views[cand_id].first];
            }
            new_center_f = new_center_f / ((float)token[j].size());
            
            // use the closest one as new center
            int minIndex = -1;
            float minDis = INFINITY;
            for(size_t token_id = 0; token_id < token[j].size(); token_id++){
              Model::Point f = view_feats[candidate_views[token[j][token_id]].first];
              float dis = a1 * (new_center_f.x - f.x) + a2 * (new_center_f.y - f.y) + a3 * (new_center_f.z - f.z);
              if(dis < minDis){
                  minDis = dis;
                  minIndex = token_id;
              }
            }
            center_ids[j] = token[j][minIndex];
          }
        }

        std::cout << "Final source views after stage 2: " << candidate_views.size();
        for(auto cand_id: center_ids){
          std::cout << candidate_views[cand_id].first << ",";
        }
        std::cout << std::endl;

        // set the src view
        for (size_t i = 0; i < k; ++i) {
          problem.src_image_idxs.push_back(candidate_views[center_ids[i]].first);
        }
      }
      else{
        // stage_mode = 1
        const size_t eff_max_num_src_images =
          std::min(candidate_views.size(), max_num_src_images);
        std::partial_sort(candidate_views.begin(),
                      candidate_views.begin() + eff_max_num_src_images,
                      candidate_views.end(),
                      [](const std::pair<int, float>& image1,
                          const std::pair<int, float>& image2) {
                        return image1.second > image2.second;
                      });
        problem.src_image_idxs.reserve(eff_max_num_src_images);
        // add the k src imgs with high scores
        for (size_t i = 0; i < eff_max_num_src_images; ++i) {
          problem.src_image_idxs.push_back(candidate_views[i].first);
        }
      }
    }
    else {
      problem.src_image_idxs.reserve(problem_config.src_image_names.size());
      for (const auto& src_image_name : problem_config.src_image_names) {
        problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
      }
    }

    if (problem.src_image_idxs.empty()) {
      std::cout
          << StringPrintf(
                 "WARNING: Ignoring reference image %s, because it has no "
                 "source images.",
                 problem_config.ref_image_name.c_str())
          << std::endl;
    } else {
      problems_.push_back(problem);
    }
  }

  std::cout << StringPrintf("Configuration has %d problems...",
                            problems_.size())
            << std::endl;
}

void PatchMatchController::ReadGpuIndices() {
  gpu_indices_ = CSVToVector<int>(options_.gpu_index);
  if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    CHECK_GT(num_cuda_devices, 0);
    gpu_indices_.resize(num_cuda_devices);
    std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);
  }
}

void PatchMatchController::ProcessProblem(const PatchMatchOptions& options,
                                          const size_t problem_idx) {
  if (IsStopped()) {
    return;
  }

  const auto& model = workspace_->GetModel();

  auto& problem = problems_.at(problem_idx);
  const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
  CHECK_GE(gpu_index, -1);

  const std::string& stereo_folder = workspace_->GetOptions().stereo_folder;
  const std::string output_type =
      options.geom_consistency ? "geometric" : "photometric";
  const std::string image_name = model.GetImageName(problem.ref_image_idx);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const std::string depth_map_path =
      JoinPaths(workspace_path_, stereo_folder, "depth_maps", file_name);
  const std::string normal_map_path =
      JoinPaths(workspace_path_, stereo_folder, "normal_maps", file_name);
  const std::string consistency_graph_path = JoinPaths(
      workspace_path_, stereo_folder, "consistency_graphs", file_name);

  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!options.write_consistency_graph ||
       ExistsFile(consistency_graph_path))) {
    return;
  }

  PrintHeading1(StringPrintf("Processing view %d / %d for %s", problem_idx + 1,
                             problems_.size(), image_name.c_str()));

  auto patch_match_options = options;

  if (patch_match_options.depth_min < 0 || patch_match_options.depth_max < 0) {
    patch_match_options.depth_min =
        depth_ranges_.at(problem.ref_image_idx).first;
    patch_match_options.depth_max =
        depth_ranges_.at(problem.ref_image_idx).second;
    CHECK(patch_match_options.depth_min > 0 &&
          patch_match_options.depth_max > 0)
        << " - You must manually set the minimum and maximum depth, since no "
           "sparse model is provided in the workspace.";
  }

  patch_match_options.gpu_index = std::to_string(gpu_index);

  if (patch_match_options.sigma_spatial <= 0.0f) {
    patch_match_options.sigma_spatial = patch_match_options.window_radius;
  }

  std::vector<Image> images = model.images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  if (options.geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
  }

  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  {
    // Collect all used images in current problem.
    std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                                            problem.src_image_idxs.end());
    used_image_idxs.insert(problem.ref_image_idx);

    patch_match_options.filter_min_num_consistent =
        std::min(static_cast<int>(used_image_idxs.size()) - 1,
                 patch_match_options.filter_min_num_consistent);

    // Only access workspace from one thread at a time and only spawn resample
    // threads from one master thread at a time.
    std::unique_lock<std::mutex> lock(workspace_mutex_);

    std::cout << "Reading inputs..." << std::endl;
    std::vector<int> src_image_idxs;
    for (const auto image_idx : used_image_idxs) {
      std::string image_path = workspace_->GetBitmapPath(image_idx);
      std::string depth_path = workspace_->GetDepthMapPath(image_idx);
      std::string normal_path = workspace_->GetNormalMapPath(image_idx);

      if (!ExistsFile(image_path) ||
          (options.geom_consistency && !ExistsFile(depth_path)) ||
          (options.geom_consistency && !ExistsFile(normal_path))) {
        if (options.allow_missing_files) {
          std::cout << StringPrintf(
                           "WARN: Skipping source image %d: %s for missing "
                           "image or depth/normal map",
                           image_idx, model.GetImageName(image_idx).c_str())
                    << std::endl;
          continue;
        } else {
          std::cout
              << StringPrintf(
                     "ERROR: Missing image or map dependency for image %d: %s",
                     image_idx, model.GetImageName(image_idx).c_str())
              << std::endl;
        }
      }

      if (image_idx != problem.ref_image_idx) {
        src_image_idxs.push_back(image_idx);
      }
      images.at(image_idx).SetBitmap(workspace_->GetBitmap(image_idx));
      if (options.geom_consistency) {
        depth_maps.at(image_idx) = workspace_->GetDepthMap(image_idx);
        normal_maps.at(image_idx) = workspace_->GetNormalMap(image_idx);
      }
    }
    problem.src_image_idxs = src_image_idxs;
  }

  problem.Print();
  patch_match_options.Print();

  PatchMatch patch_match(patch_match_options, problem);
  patch_match.Run();

  std::cout << std::endl
            << StringPrintf("Writing %s output for %s", output_type.c_str(),
                            image_name.c_str())
            << std::endl;

  patch_match.GetDepthMap().Write(depth_map_path);
  patch_match.GetNormalMap().Write(normal_map_path);
  if (options.write_consistency_graph) {
    patch_match.GetConsistencyGraph().Write(consistency_graph_path);
  }
}

}  // namespace mvs
}  // namespace colmap
