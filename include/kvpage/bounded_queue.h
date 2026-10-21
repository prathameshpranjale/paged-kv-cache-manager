#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace kvp {

// Bounded multi-producer / multi-consumer queue (mutex + condition variables).
template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : cap_(capacity ? capacity : 1) {}

    // Blocks while full. Returns false if the queue was closed (item not queued).
    bool push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
        return true;
    }
    // Blocks while empty. Returns false once closed and drained.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    std::deque<T> q_;
    std::size_t cap_;
    bool closed_ = false;
};

}  // namespace kvp
