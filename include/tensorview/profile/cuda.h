// Copyright 2021 Yan Yan
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once 

#ifdef TV_CUDA
#include <cuda_runtime_api.h>
#endif 
#include <tensorview/tensor.h>
#include <unordered_map>
#include <memory>
#include <string>
#include <chrono>
namespace tv {

#ifdef TV_CUDA

class CUDAEventCore {
private:
cudaEvent_t event_ = 0;
public:

CUDAEventCore(){
    checkCudaErrors(cudaEventCreate(&event_));
}

~CUDAEventCore(){
    if (event_){{
        cudaEventDestroy(event_);
    }}
}

void record(cudaStream_t stream = nullptr){
    checkCudaErrors(cudaEventRecord(event_, stream));
}

void sync(cudaStream_t stream = nullptr){
    checkCudaErrors(cudaEventSynchronize(event_));
}

cudaEvent_t get_event(){
    return event_;
}

};

class CUDAEvent {
private:
std::shared_ptr<CUDAEventCore> event_ = 0;
public:
std::string name;

CUDAEvent(std::string name): event_(std::make_shared<CUDAEventCore>()), name(name){
}

void record(std::uintptr_t stream = 0){
    TV_ASSERT_RT_ERR(event_, "event is empty");
    event_->record(reinterpret_cast<cudaStream_t>(stream));
}

void sync(std::uintptr_t stream = 0){
    TV_ASSERT_RT_ERR(event_, "event is empty");
    event_->sync();
}

static float duration(CUDAEvent start, CUDAEvent stop){
    float ms;
    TV_ASSERT_RT_ERR(start.event_, "event is empty");
    TV_ASSERT_RT_ERR(stop.event_, "event is empty");
    checkCudaErrors(cudaEventElapsedTime(&ms, start.event_->get_event(), stop.event_->get_event()));
    return ms;
}

};
#else


class CUDAEvent {
private:
std::chrono::time_point<std::chrono::steady_clock> cur_time_;
public:
std::string name;

CUDAEvent(std::string name): name(name){
}

void record(std::uintptr_t stream = 0){
    cur_time_ = std::chrono::steady_clock::now();
}

void sync(std::uintptr_t stream = 0){
}

static float duration(CUDAEvent start, CUDAEvent stop){
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(
          stop.cur_time_ - start.cur_time_).count() / 1000.0f;
    return ms;
}

};
#endif

class CUDAKernelTimerCore {
private:
std::vector<std::string> namespaces_;
std::unordered_map<std::string, CUDAEvent> name_to_event_;
std::unordered_map<std::string, std::pair<std::string, std::string>> name_to_pair_;
std::vector<std::string> names_;
public:

void push(std::string name) noexcept {
    namespaces_.push_back(name);
}
void pop() {
    TV_ASSERT_RT_ERR(!namespaces_.empty(), "you pop value from empty namespaces");
    namespaces_.pop_back();
}

std::string get_current_namespace() {
    if (namespaces_.empty()){{
        return "";
    }}
    std::string res = namespaces_[0];
    for (int i = 1; i < namespaces_.size(); ++i){{
        res += "." + namespaces_[i];
    }}
    return res;
}

std::string add_namespace_to_name(std::string name) {
    auto unique_name = name;
    auto cur_ns = get_current_namespace();
    if (name.empty()){{
        return cur_ns;
    }}
    if (cur_ns.size() > 0){{
        unique_name = cur_ns + "." + name;
    }}
    return unique_name;
}

void record(std::string name, std::uintptr_t stream = 0){
    auto unique_name = add_namespace_to_name(name);
    TV_ASSERT_RT_ERR(name_to_event_.find(unique_name) == name_to_event_.end(), "your name", unique_name, "already exists");
    CUDAEvent newev(unique_name);
    newev.record(stream);
    name_to_event_.insert({{unique_name, newev}});
    names_.push_back(unique_name);
}

bool has_pair(std::string name){
    name = add_namespace_to_name(name);
    return name_to_pair_.find(name) != name_to_pair_.end();
}

void sync_all_event() {
    for (auto& p : name_to_event_){{
        p.second.sync();
    }}
}

void insert_pair(std::string name, std::string start, std::string stop){
    name = add_namespace_to_name(name);
    start = add_namespace_to_name(start);
    stop = add_namespace_to_name(stop);
    TV_ASSERT_RT_ERR(name_to_pair_.find(name) == name_to_pair_.end(), "your name", name, "already exists");
    name_to_pair_[name] = {start, stop};
}

std::unordered_map<std::string, float> get_all_pair_duration(){
    std::unordered_map<std::string, float> res;
    sync_all_event();
    for (auto& p : name_to_pair_){{
        auto& ev_start = name_to_event_.at(p.second.first);
        auto& ev_stop = name_to_event_.at(p.second.second);
        res[p.first] = CUDAEvent::duration(ev_start, ev_stop);
    }}
    return res;
}

};


class CUDAKernelTimer {
private:
std::shared_ptr<CUDAKernelTimerCore> timer_ptr_;
bool enable_;

public:
CUDAKernelTimer(bool enable): timer_ptr_(std::make_shared<CUDAKernelTimerCore>()), enable_(enable){}

void push(std::string name) {
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->push(name);
    }}
}
bool enable(){
    return enable_;
}

void pop() {
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->pop();
    }}
}


void record(std::string name, std::uintptr_t stream = 0){
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->record(name, stream);
    }}
}
void insert_pair(std::string name, std::string start, std::string stop){
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        return timer_ptr_->insert_pair(name, start, stop);
    }}
}

bool has_pair(std::string name){
    if (!enable_){
        return false;
    }
    TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
    return timer_ptr_->has_pair(name);
}

void sync_all_event() {
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        timer_ptr_->sync_all_event();
    }}
}

std::unordered_map<std::string, float> get_all_pair_duration(){
    std::unordered_map<std::string, float> res;
    if (enable_){{
        TV_ASSERT_RT_ERR(timer_ptr_, "event is empty");
        return timer_ptr_->get_all_pair_duration();
    }}
    return res;
}

};

#ifdef TV_CUDA
class CUDAKernelTimerGuard {
private:
std::string name_;
CUDAKernelTimer timer_;
cudaStream_t stream_ = nullptr;
public:

CUDAKernelTimerGuard(std::string name, CUDAKernelTimer timer, cudaStream_t stream = nullptr): name_(name),  timer_(timer),  stream_(stream){
    if (timer_.enable()){{
        if (!name_.empty()){
            timer_.push(name_);
        }
        timer_.insert_pair("", "start", "stop");
        timer_.record("start", reinterpret_cast<std::uintptr_t>(stream_));
    }}

}

~CUDAKernelTimerGuard(){
    if (timer_.enable()){{
        timer_.record("stop", reinterpret_cast<std::uintptr_t>(stream_));
        if (!name_.empty()){
            timer_.pop();
        }
    }}
}

};
#endif

}
