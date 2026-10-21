#include "runtime/sentences.hpp"

#include <iostream>
#include <stdexcept>

#include "python/python.hpp"
#include "runtime/interp.hpp"
#include "safetensors/reader.hpp"
#include "safetensors/writer.hpp"

namespace milo::runtime {
namespace {

/// The text of a value, for the messages a sentence prints.
std::string textOf(const Value& value) { return value.toDisplayString(); }

}  // namespace

const Value* SentenceValues::slot(const std::string& name) const {
  for (const auto& entry : slots) {
    if (entry.first == name) return &entry.second;
  }
  return nullptr;
}

const Value* SentenceValues::option(const std::string& key) const {
  for (const auto& entry : options) {
    if (entry.first == key) return &entry.second;
  }
  return nullptr;
}

void executeSentence(Interpreter& interpreter, const std::string& verb, SourceSpan span,
                     const SentenceValues& values, const std::shared_ptr<Environment>& env) {
  std::ostream& out = interpreter.out();
  std::ostream& notes = interpreter.notes();
  if (verb == "load") {
    const Value* kindValue = values.slot("kind");
    const Value* pathValue = values.slot("path");
    const Value* nameValue = values.slot("name");
    if (kindValue == nullptr || pathValue == nullptr || nameValue == nullptr) {
      throw RuntimeError{"this `load` sentence is missing something it needs", span, {}};
    }

    const std::string kind = textOf(*kindValue);
    const std::string path = textOf(*pathValue);
    const std::string name = textOf(*nameValue);

    if (kind == "data" || kind == "dataset" || kind == "tokenizer") {
      env->define(name, Value::dataset(DatasetValue{name, path, kind}));
    } else {
      ModelValue model;
      model.name = name;
      model.path = path;
      model.kind = kind;

      // A model is a safetensors file. The tensors are mapped straight
      // from disk: nothing is copied into memory.
      try {
        auto file = std::make_shared<safetensors::File>(safetensors::File::open(path));
        auto tensors =
            std::make_shared<std::map<std::string, std::shared_ptr<tensor::Tensor>>>();
        for (const std::string& tensorName : file->names()) {
          (*tensors)[tensorName] = file->tensor(tensorName);
        }
        model.tensors = tensors;
        model.metadata = file->header().metadata;
      } catch (const safetensors::Error& error) {
        throw RuntimeError{"I could not load '" + path + "': " + error.what(), span, {}};
      }

      env->define(name, Value::model(std::move(model)));
    }
    out << "loaded " << kind << " '" << name << "' from '" << path << "'\n";
    return;
  }

  if (verb == "split") {
    const Value* sourceValue = values.slot("source");
    const Value* leftValue = values.slot("left");
    const Value* rightValue = values.slot("right");
    if (sourceValue == nullptr || leftValue == nullptr || rightValue == nullptr) {
      throw RuntimeError{"this `split` sentence is missing something it needs", span, {}};
    }
    const std::string source = textOf(*sourceValue);
    const std::string left = textOf(*leftValue);
    const std::string right = textOf(*rightValue);

    std::int64_t percent = 90;
    if (const Value* ratio = values.slot("ratio")) {
      percent = ratio->kind() == Value::Kind::Int ? ratio->asInt() : 90;
    }

    env->define(left, Value::dataset(DatasetValue{left, source, "dataset"}));
    env->define(right, Value::dataset(DatasetValue{right, source, "dataset"}));
    out << "split '" << source << "' into '" << left << "' (" << percent << "%) and '" << right
        << "' (" << (100 - percent) << "%)\n";
    return;
  }

  if (verb == "use") {
    const Value* moduleValue = values.slot("module");
    const Value* nameValue = values.slot("name");
    if (moduleValue == nullptr || nameValue == nullptr) {
      throw RuntimeError{"this `use` sentence is missing something it needs", span, {}};
    }
    const std::string module = textOf(*moduleValue);
    const std::string alias = textOf(*nameValue);

    // The Python bridge is optional: a build without it says so.
    if (!python::Context::isAvailable()) {
      throw RuntimeError{"this build has no Python in it", span, {}};
    }
    Value imported = interpreter.pythonContext().importModule(module);
    env->define(alias, imported);
    notes << "note: imported the Python module '" << module << "' as '" << alias << "'\n";
    return;
  }

  if (verb == "show") {
    const Value* metricValue = values.slot("metric");
    if (metricValue == nullptr) {
      throw RuntimeError{"this `show` sentence is missing the thing to show", span, {}};
    }
    const std::string metric = textOf(*metricValue);
    notes << "note: '" << metric << "' is not computed yet; evaluation arrives in a later phase\n";
    return;
  }

  if (verb == "save") {
    const Value* modelValue = values.slot("model");
    const Value* pathValue = values.slot("path");
    if (modelValue == nullptr || pathValue == nullptr) {
      throw RuntimeError{"this `save` sentence is missing something it needs", span, {}};
    }
    const std::string modelName = textOf(*modelValue);
    const std::string path = textOf(*pathValue);

    const Value* model = env->find(modelName);
    if (model == nullptr) {
      throw RuntimeError{"I cannot find anything called '" + modelName + "'", span, {}};
    }
    if (model->kind() != Value::Kind::Model) {
      throw RuntimeError{"'" + modelName + "' is not a model", span, {}};
    }
    const auto& value = model->asModel();
    if (!value.tensors) {
      throw RuntimeError{"'" + modelName + "' has no tensors to save", span, {}};
    }

    try {
      safetensors::Writer writer(path);
      for (const auto& entry : *value.tensors) {
        writer.add(entry.first, *entry.second);
      }
      for (const auto& entry : value.metadata) {
        writer.setMetadata(entry.first, entry.second);
      }
      writer.write();
    } catch (const safetensors::Error& error) {
      throw RuntimeError{"I could not save '" + path + "': " + error.what(), span, {}};
    }

    out << "saved model '" << modelName << "' to '" << path << "'\n";
    return;
  }

  if (verb == "quantize" || verb == "fine-tune" || verb == "train" ||
      verb == "evaluate" || verb == "predict") {
    notes << "note: '" << verb
           << "' is recognised and parsed correctly, but it does not do any work yet\n";
    return;
  }

  throw RuntimeError{"I do not know how to carry out '" + verb + "'", span, {}};
}

}  // namespace milo::runtime
