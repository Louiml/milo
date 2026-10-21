// Runtime values.
#ifndef MILO_RUNTIME_VALUE_HPP
#define MILO_RUNTIME_VALUE_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/ast.hpp"
#include "autograd/state.hpp"
#include "tensor/quantization.hpp"
#include "tensor/tensor.hpp"

namespace milo {

class Interpreter;

namespace runtime {

class Environment;
class Value;

/// A model or dataset placeholder.
///
/// Phase 1 recorded what was loaded; now a model can hold the tensors of
/// a safetensors file, mapped straight from disk.
struct ModelValue {
  std::string name;
  std::string path;
  /// `model`, `weights`, or `adapter`.
  std::string kind;
  /// The tensors of the model, when it was loaded from a file.
  std::shared_ptr<std::map<std::string, std::shared_ptr<tensor::Tensor>>> tensors;
  /// The metadata of the file it came from.
  std::vector<std::pair<std::string, std::string>> metadata;
  /// The total number of parameters across the tensors.
  std::int64_t parameterCount() const;
};

struct DatasetValue {
  std::string name;
  std::string path;
  /// `data`, `dataset`, or `tokenizer`.
  std::string kind;
};

/// A Python module imported into a Milo program.
///
/// A Python object that Milo is holding on to.
///
/// The object is kept alive by a `shared_ptr` whose deleter knows how to
/// give the reference back, so the runtime does not need to know anything
/// about Python.
struct PythonObjectValue {
  /// What the object is: a module, or something else.
  enum class Kind {
    Module,
    Object,
  };

  /// The module's name, for the display. Empty for other objects.
  std::string name;
  Kind kind = Kind::Object;
  /// The Python object itself.
  std::shared_ptr<void> handle;
};

/// A callable that does nothing yet.
struct StubValue {
  std::string name;
};

/// A layer: a group of named parameters with a forward pass.
struct ModuleValue {
  /// `linear`, `layer_norm`, `lora`, ...
  std::string kind;
  /// The parameters the optimiser should update.
  std::map<std::string, std::shared_ptr<tensor::Tensor>> parameters;
  /// The parameters that are trained but not updated.
  std::map<std::string, std::shared_ptr<tensor::Tensor>> buffers;
  /// The base weight, when it is quantised.
  std::shared_ptr<tensor::QuantizedTensor> quantizedWeight;
};

/// An optimiser: it holds the parameters and knows how to update them.
struct OptimizerValue {
  /// `sgd`, `adam`, `adamw`.
  std::string kind;
  std::vector<std::shared_ptr<tensor::Tensor>> parameters;
  double learning_rate = 0.001;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  double weight_decay = 0.0;
  /// The moments and the step count. This is shared, so that a copy of
  /// the optimiser — which is what a method call makes — still sees the
  /// same state.
  std::shared_ptr<autograd::OptimizerState> state =
      std::make_shared<autograd::OptimizerState>();
};

/// An iterator over a list of values.
///
/// The bytecode machine walks a list with one of these. The tree-walking
/// evaluator does not need one, because it walks the list directly.
struct IteratorValue {
  /// The values being walked over.
  std::shared_ptr<std::vector<Value>> items;
  /// How far along we are.
  std::size_t index = 0;
};

/// A user-defined function.
struct FunctionValue;

/// A builtin implemented in C++.
struct BuiltinValue;

/// A Milo value.
///
/// The representation is deliberately simple: one tag plus the payload.
/// Phase 2 replaces this with a `std::variant` once the tensor handle is
/// settled; nothing above this layer depends on the representation.
class Value {
 public:
  enum class Kind {
    Null,
    Bool,
    Int,
    Float,
    String,
    List,
    Dict,
    Model,
    Dataset,
    Python,
    Module,
    Stub,
    Tensor,
    Quantized,
    Optimizer,
    Iterator,
    MachineFunction,
    Function,
    Builtin,
  };

  using List = std::vector<Value>;
  using Dict = std::vector<std::pair<std::string, Value>>;

  Value() = default;

  static Value null();
  static Value boolean(bool value);
  static Value integer(std::int64_t value);
  static Value floating(double value);
  static Value string(std::string value);
  static Value list(List value);
  static Value dict(Dict value);
  static Value model(ModelValue value);
  static Value dataset(DatasetValue value);
  static Value pythonModule(PythonObjectValue value);
  static Value pythonObject(PythonObjectValue value);
  static Value layer(ModuleValue value);
  static Value optimizer(OptimizerValue value);
  static Value stub(StubValue value);
  static Value tensor(std::shared_ptr<tensor::Tensor> value);
  static Value quantized(std::shared_ptr<tensor::QuantizedTensor> value);
  static Value iterator(std::shared_ptr<IteratorValue> value);
  /// A function compiled by the bytecode machine.
  ///
  /// The payload is erased, so that the runtime does not have to know
  /// about the machine. Only the machine itself can call one.
  static Value machineFunction(std::shared_ptr<void> value);
  static Value function(std::shared_ptr<FunctionValue> value);
  static Value builtin(std::shared_ptr<BuiltinValue> value);

  Kind kind() const { return kind_; }
  bool isNull() const { return kind_ == Kind::Null; }
  bool isTruthy() const;

  bool asBool() const { return bool_; }
  std::int64_t asInt() const { return int_; }
  double asFloat() const { return float_; }
  const std::string& asString() const { return string_; }
  const List& asList() const { return *list_; }
  List& asList() { return *list_; }
  const Dict& asDict() const { return *dict_; }
  Dict& asDict() { return *dict_; }
  const ModelValue& asModel() const { return *model_; }
  const DatasetValue& asDataset() const { return *dataset_; }
  const PythonObjectValue& asPythonObject() const { return *python_object_; }
  const ModuleValue& asModule() const { return *module_; }
  OptimizerValue& asOptimizer() { return *optimizer_; }
  const OptimizerValue& asOptimizer() const { return *optimizer_; }
  const StubValue& asStub() const { return *stub_; }
  const std::shared_ptr<tensor::Tensor>& asTensor() const { return tensor_; }
  const std::shared_ptr<tensor::QuantizedTensor>& asQuantized() const { return quantized_; }
  const std::shared_ptr<IteratorValue>& asIterator() const { return iterator_; }
  const std::shared_ptr<void>& machineFunction() const { return machine_function_; }
  const std::shared_ptr<FunctionValue>& asFunction() const { return function_; }
  const std::shared_ptr<BuiltinValue>& asBuiltin() const { return builtin_; }

  /// The name of the value's type, as `type(x)` reports it.
  const char* typeName() const;

  /// How the value prints.
  std::string toDisplayString() const;

  /// How the value looks inside a list or a dictionary.
  std::string toReprString() const;

  /// Look up a key in a dictionary.
  const Value* find(const std::string& key) const;

 private:
  Kind kind_ = Kind::Null;
  bool bool_ = false;
  std::int64_t int_ = 0;
  double float_ = 0.0;
  std::string string_;
  std::shared_ptr<List> list_;
  std::shared_ptr<Dict> dict_;
  std::shared_ptr<ModelValue> model_;
  std::shared_ptr<DatasetValue> dataset_;
  std::shared_ptr<PythonObjectValue> python_object_;
  std::shared_ptr<ModuleValue> module_;
  std::shared_ptr<OptimizerValue> optimizer_;
  std::shared_ptr<StubValue> stub_;
  std::shared_ptr<tensor::Tensor> tensor_;
  std::shared_ptr<tensor::QuantizedTensor> quantized_;
  std::shared_ptr<IteratorValue> iterator_;
  std::shared_ptr<void> machine_function_;
  std::shared_ptr<FunctionValue> function_;
  std::shared_ptr<BuiltinValue> builtin_;
};

/// A user-defined function.
struct FunctionValue {
  std::string name;
  std::vector<std::string> params;
  /// The scope the function was written in.
  ///
  /// This is a weak reference on purpose. A function is usually defined in
  /// the scope it closes over, and that scope holds the function: the two
  /// would keep each other alive for ever, and the scope would never be
  /// given back.
  std::weak_ptr<Environment> closure;
  /// The scope, kept alive for a function that has left it.
  ///
  /// A function returned out of the scope it was written in still needs
  /// that scope, so it takes it with it. The scope has let go of the
  /// function by then, so the two do not hold each other up.
  std::shared_ptr<Environment> keepAlive;
  std::vector<ast::StmtPtr> body;
  std::shared_ptr<ast::TypeRef> return_type;
};

/// A builtin implemented in C++.
struct BuiltinValue {
  std::string name;
  std::function<Value(Interpreter&, const std::vector<Value>&,
                      const std::vector<std::pair<std::string, Value>>&)>
      fn;
};

}  // namespace runtime
}  // namespace milo

#endif  // MILO_RUNTIME_VALUE_HPP
