#include "autograd/tape.hpp"

#include <algorithm>
#include <stdexcept>

#include "tensor/ops.hpp"

namespace milo::autograd {
namespace {

thread_local Tape g_tape;
thread_local bool g_recording = true;

}  // namespace

Tape& Tape::current() { return g_tape; }

void Tape::clear() { nodes_.clear(); }

void Tape::push(Node node) { nodes_.push_back(std::move(node)); }

void Tape::backward(const std::shared_ptr<tensor::Tensor>& root) {
  if (!root) return;

  if (root->shape().rank() != 0) {
    throw std::invalid_argument("a backward pass can only start from a single number");
  }

  // The seed: the gradient of a scalar with respect to itself is one.
  root->setGrad(std::make_shared<tensor::Tensor>(
      tensor::Tensor::ones(root->dtype(), root->shape())));

  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    if (it->backward) it->backward();
  }

  // The graph has been used; the gradients on the leaves stay.
  nodes_.clear();
}

bool recording() { return g_recording; }

void setRecording(bool value) { g_recording = value; }

}  // namespace milo::autograd
