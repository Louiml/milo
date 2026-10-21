// The gradient tape.
//
// Autograd is a *runtime facility*, not a language feature: the tensor
// engine has no idea it exists. The layer above records what it did, and
// the tape remembers enough to undo it.
#ifndef MILO_AUTOGRAD_TAPE_HPP
#define MILO_AUTOGRAD_TAPE_HPP

#include <functional>
#include <memory>
#include <vector>

#include "tensor/tensor.hpp"

namespace milo::autograd {

/// One step of the graph.
///
/// A node remembers the tensors it read, the tensors it produced, and how
/// to push the output's gradient back into the inputs'.
struct Node {
  std::vector<std::shared_ptr<tensor::Tensor>> inputs;
  std::vector<std::shared_ptr<tensor::Tensor>> outputs;
  std::function<void()> backward;
};

/// The graph of the current forward pass.
class Tape {
 public:
  /// The tape for this thread.
  static Tape& current();

  /// Forget the graph.
  void clear();

  /// How many steps the graph has.
  std::size_t size() const { return nodes_.size(); }
  bool empty() const { return nodes_.empty(); }

  /// Add a step.
  void push(Node node);

  /// Run the graph backwards from `root`.
  ///
  /// `root` must be a scalar: its gradient starts at one. The graph is
  /// consumed, so a second backward pass has nothing to do.
  void backward(const std::shared_ptr<tensor::Tensor>& root);

 private:
  std::vector<Node> nodes_;
};

/// Is recording turned on?
///
/// Recording is off inside `no_grad`, and it is off by default for tensors
/// that do not ask for a gradient.
bool recording();

/// Turn recording on or off.
void setRecording(bool value);

/// Run `body` without recording anything.
template <typename Fn>
auto withoutRecording(Fn&& body) -> decltype(body()) {
  const bool previous = recording();
  setRecording(false);
  struct Restore {
    bool previous;
    ~Restore() { setRecording(previous); }
  } restore{previous};
  return body();
}

}  // namespace milo::autograd

#endif  // MILO_AUTOGRAD_TAPE_HPP
