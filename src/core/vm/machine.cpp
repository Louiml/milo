#include "core/vm/machine.hpp"

#include <algorithm>
#include <iostream>

#include "python/python.hpp"
#include "runtime/sentences.hpp"

namespace milo::vm {
namespace {

/// The deepest a call may go.
constexpr int kMaxCallDepth = 512;

/// The name of the operator an instruction stands for.
const char* operatorName(Op op) {
  switch (op) {
    case Op::Add: return "+";
    case Op::Sub: return "-";
    case Op::Mul: return "*";
    case Op::Div: return "/";
    case Op::Mod: return "%";
    case Op::Pow: return "**";
    case Op::Eq: return "==";
    case Op::Ne: return "!=";
    case Op::Lt: return "<";
    case Op::Le: return "<=";
    case Op::Gt: return ">";
    case Op::Ge: return ">=";
    default: return "+";
  }
}

/// Hand a scope over to a function that is leaving it.
///
/// A function closes over the scope it was written in, and that scope
/// holds the function: the two keep each other alive for ever, and the
/// scope is never given back. When the function leaves, it takes the scope
/// with it and the scope lets go of the function, so that neither of them
/// is kept alive by the other.
void releaseReturnedFunction(const runtime::Value& value,
                             const std::shared_ptr<runtime::Environment>& scope) {
  if (value.kind() != runtime::Value::Kind::MachineFunction) return;
  auto* function = static_cast<Function*>(value.machineFunction().get());
  auto closed = function->closure.lock();
  if (!closed || closed.get() != scope.get()) return;
  function->keepAlive = closed;
  scope->release(function->name);
}

}  // namespace

void Machine::push(runtime::Value value) { stack_.push_back(std::move(value)); }

runtime::Value Machine::pop() {
  if (stack_.empty()) {
    throw RuntimeError{"the stack is empty", SourceSpan::start(), {}};
  }
  runtime::Value value = std::move(stack_.back());
  stack_.pop_back();
  return value;
}

runtime::Value& Machine::peek() {
  if (stack_.empty()) {
    throw RuntimeError{"the stack is empty", SourceSpan::start(), {}};
  }
  return stack_.back();
}

runtime::Value Machine::run(const Program& program) {
  program_ = &program;
  stack_.clear();
  frames_.clear();
  depth_ = 0;

  Frame frame;
  // The machine shares the interpreter's global scope, so that the
  // builtins are visible.
  frame.env = interpreter_.globals();
  frames_.push_back(frame);

  currentChunk_ = &program.main;
  currentIp_ = 0;

  while (true) {
    if (currentIp_ >= currentChunk_->code.size()) {
      throw RuntimeError{"the program ran off the end of its instructions", SourceSpan::start(), {}};
    }
    const Instruction instruction = currentChunk_->code[currentIp_++];
    const Chunk* chunk = currentChunk_;

    switch (instruction.op) {
      case Op::Nop:
        break;
      case Op::Constant:
        push(chunk->constants[instruction.a]);
        break;
      case Op::Null:
        push(runtime::Value::null());
        break;
      case Op::True:
        push(runtime::Value::boolean(true));
        break;
      case Op::False:
        push(runtime::Value::boolean(false));
        break;
      case Op::Pop:
        pop();
        break;
      case Op::Dup:
        push(peek());
        break;

      case Op::LoadName: {
        const std::string& name = chunk->names[instruction.a];
        const runtime::Value* value = frames_.back().env->find(name);
        if (value == nullptr) {
          interpreter_.raiseUnknownName(name, SourceSpan::start());
        }
        push(*value);
        break;
      }
      case Op::StoreName: {
        const std::string& name = chunk->names[instruction.a];
        runtime::Value value = pop();
        if (!frames_.back().env->assign(name, value)) {
          interpreter_.raiseUnknownName(name, SourceSpan::start());
        }
        break;
      }
      case Op::DefineName: {
        const std::string& name = chunk->names[instruction.a];
        runtime::Value value = pop();
        runtime::Environment& env = *frames_.back().env;
        if (env.findLocal(name) != nullptr && !env.allowRedeclaration()) {
          interpreter_.raise("'" + name + "' is already declared in this scope",
                             SourceSpan::start());
        }
        env.define(name, value);
        break;
      }
      case Op::DefineVar: {
        // A loop's variable replaces whatever was there.
        const std::string& name = chunk->names[instruction.a];
        runtime::Value value = pop();
        frames_.back().env->define(name, value);
        break;
      }
      case Op::AllowRedeclaration:
        frames_.back().env->setAllowRedeclaration(true);
        break;
      case Op::DisallowRedeclaration:
        frames_.back().env->setAllowRedeclaration(false);
        break;
      case Op::GetAttr: {
        runtime::Value target = peek();
        const runtime::Value field =
            interpreter_.memberOperation(target, chunk->names[instruction.a], SourceSpan::start());
        stack_.back() = field;
        break;
      }

      case Op::Call: {
        const std::uint32_t count = instruction.a;
        std::vector<runtime::Value> args;
        args.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          args.push_back(pop());
        }
        std::reverse(args.begin(), args.end());

        // The keyword arguments, if there are any.
        std::vector<std::pair<std::string, runtime::Value>> kwargs;
        if (instruction.b != 0) {
          const std::vector<std::string>& names = chunk->keywordGroups[instruction.b - 1];
          const std::size_t positional = args.size() - names.size();
          for (std::size_t i = 0; i < names.size(); ++i) {
            kwargs.emplace_back(names[i], args[positional + i]);
          }
          args.resize(positional);
        }

        runtime::Value callee = pop();

        if (callee.kind() == runtime::Value::Kind::MachineFunction) {
          const auto function = std::static_pointer_cast<Function>(callee.machineFunction());
          enterFunction(function, args);
          break;
        }

        runtime::Value result = runtime::Value::null();
        if (!interpreter_.callSpecialValue(callee, args, kwargs, result)) {
          interpreter_.raise("this is not something I can call", SourceSpan::start());
        }
        push(result);
        break;
      }
      case Op::Return: {
        runtime::Value result = pop();
        // A function that leaves the scope it was written in takes the
        // scope with it, and the scope lets go of the function, so that
        // neither of them is kept alive by the other.
        releaseReturnedFunction(result, frames_.back().env);
        if (frames_.size() == 1) {
          return result;
        }
        const Frame finished = frames_.back();
        frames_.pop_back();
        stack_.resize(finished.stackBase);
        currentChunk_ = finished.chunk;
        currentIp_ = finished.returnAddress;
        --depth_;
        push(result);
        break;
      }

      case Op::Jump:
        currentIp_ = instruction.a;
        break;
      case Op::JumpIfFalse: {
        runtime::Value value = pop();
        if (!value.isTruthy()) currentIp_ = instruction.a;
        break;
      }
      case Op::JumpIfFalseKeep: {
        if (!peek().isTruthy()) currentIp_ = instruction.a;
        break;
      }
      case Op::JumpIfTrue: {
        runtime::Value value = pop();
        if (value.isTruthy()) currentIp_ = instruction.a;
        break;
      }
      case Op::JumpIfTrueKeep: {
        if (peek().isTruthy()) currentIp_ = instruction.a;
        break;
      }

      case Op::GetIter: {
        runtime::Value iterable = pop();
        if (iterable.kind() != runtime::Value::Kind::List) {
          interpreter_.raiseWrongType("I can only loop over a list", iterable, SourceSpan::start());
        }
        auto iterator = std::make_shared<runtime::IteratorValue>();
        iterator->items = std::make_shared<std::vector<runtime::Value>>(iterable.asList());
        push(runtime::Value::iterator(iterator));
        break;
      }
      case Op::ForIter: {
        runtime::Value& iteratorValue = peek();
        if (iteratorValue.kind() != runtime::Value::Kind::Iterator) {
          throw RuntimeError{"this is not an iterator", SourceSpan::start(), {}};
        }
        const auto& iterator = iteratorValue.asIterator();
        if (iterator->index >= iterator->items->size()) {
          pop();
          currentIp_ = instruction.a;
          break;
        }
        const runtime::Value item = (*iterator->items)[iterator->index++];
        push(item);
        break;
      }

      case Op::Add:
      case Op::Sub:
      case Op::Mul:
      case Op::Div:
      case Op::Mod:
      case Op::Pow:
      case Op::Eq:
      case Op::Ne:
      case Op::Lt:
      case Op::Le:
      case Op::Gt:
      case Op::Ge: {
        runtime::Value rhs = pop();
        runtime::Value lhs = pop();
        push(interpreter_.binaryOperation(operatorName(instruction.op), lhs, rhs,
                                         SourceSpan::start()));
        break;
      }
      case Op::Neg: {
        runtime::Value value = pop();
        push(interpreter_.unaryOperation("-", value, SourceSpan::start()));
        break;
      }
      case Op::Not: {
        runtime::Value value = pop();
        push(interpreter_.unaryOperation("not", value, SourceSpan::start()));
        break;
      }

      case Op::Index: {
        const std::uint32_t count = instruction.a;
        std::vector<runtime::Value> indices;
        indices.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          indices.push_back(pop());
        }
        std::reverse(indices.begin(), indices.end());
        runtime::Value target = pop();
        push(interpreter_.indexOperation(target, indices, SourceSpan::start()));
        break;
      }
      case Op::SetIndex: {
        runtime::Value value = pop();
        runtime::Value index = pop();
        runtime::Value target = pop();
        if (target.kind() != runtime::Value::Kind::Dict) {
          interpreter_.raise("I can only assign into a dictionary", SourceSpan::start());
        }
        runtime::Value::Dict entries = target.asDict();
        const std::string key = index.toDisplayString();
        bool replaced = false;
        for (auto& entry : entries) {
          if (entry.first == key) {
            entry.second = value;
            replaced = true;
            break;
          }
        }
        if (!replaced) {
          entries.emplace_back(key, value);
        }
        push(runtime::Value::dict(std::move(entries)));
        break;
      }

      case Op::Concat: {
        runtime::Value rhs = pop();
        runtime::Value lhs = pop();
        push(runtime::Value::string(lhs.toDisplayString() + rhs.toDisplayString()));
        break;
      }
      case Op::MakeList: {
        const std::uint32_t count = instruction.a;
        runtime::Value::List items;
        items.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
          items.push_back(pop());
        }
        std::reverse(items.begin(), items.end());
        push(runtime::Value::list(std::move(items)));
        break;
      }
      case Op::MakeDict: {
        const std::uint32_t count = instruction.a;
        runtime::Value::Dict entries;
        for (std::uint32_t i = 0; i < count; ++i) {
          runtime::Value value = pop();
          runtime::Value key = pop();
          entries.emplace_back(key.toDisplayString(), value);
        }
        std::reverse(entries.begin(), entries.end());
        push(runtime::Value::dict(std::move(entries)));
        break;
      }
      case Op::TypeTest: {
        runtime::Value value = pop();
        push(interpreter_.typeTestOperation(value, chunk->names[instruction.a]));
        break;
      }

      case Op::Closure: {
        // Each definition gets its own copy of the function, so that its
        // closure is its own.
        auto function = std::make_shared<Function>(*program_->functions[instruction.a]);
        function->closure = frames_.back().env;
        push(runtime::Value::machineFunction(function));
        break;
      }

      case Op::Import: {
        // Import a Python module and push it.
        const std::string& name = chunk->names[instruction.a];
        try {
          push(interpreter_.pythonContext().importModule(name));
        } catch (const std::exception& error) {
          throw RuntimeError{error.what(), SourceSpan::start(), {}};
        }
        break;
      }

      case Op::Sentence: {
        const CompiledSentence& sentence = program_->sentences[instruction.a];
        const std::size_t slotCount = instruction.b;
        const std::size_t optionCount = sentence.optionKeys.size();

        runtime::SentenceValues values;

        // The options come off the stack first: they were pushed last.
        std::vector<runtime::Value> optionValues;
        optionValues.reserve(optionCount);
        for (std::size_t i = 0; i < optionCount; ++i) {
          optionValues.push_back(pop());
        }
        std::reverse(optionValues.begin(), optionValues.end());
        for (std::size_t i = 0; i < optionCount; ++i) {
          values.options.emplace_back(sentence.optionKeys[i], optionValues[i]);
        }

        std::vector<runtime::Value> slotValues;
        slotValues.reserve(slotCount);
        for (std::size_t i = 0; i < slotCount; ++i) {
          slotValues.push_back(pop());
        }
        std::reverse(slotValues.begin(), slotValues.end());
        for (std::size_t i = 0; i < slotCount; ++i) {
          values.slots.emplace_back(sentence.slotNames[i], slotValues[i]);
        }

        runtime::executeSentence(interpreter_, sentence.verb, SourceSpan::start(), values,
                                frames_.back().env);
        break;
      }
    }
  }
}

void Machine::enterFunction(const std::shared_ptr<Function>& function,
                            const std::vector<runtime::Value>& args) {
  if (depth_ >= kMaxCallDepth) {
    throw RuntimeError{"this call went too deep; is a function calling itself without end?",
                       SourceSpan::start(), {}};
  }

  Frame frame;
  frame.env = std::make_shared<runtime::Environment>(function->closure.lock());
  for (std::size_t i = 0; i < function->params.size() && i < args.size(); ++i) {
    frame.env->define(function->params[i], args[i]);
  }
  // The function can call itself.
  frame.env->define(function->name, runtime::Value::machineFunction(function));

  frame.stackBase = stack_.size();
  frame.returnAddress = currentIp_;
  frame.chunk = currentChunk_;
  frames_.push_back(frame);
  ++depth_;

  currentChunk_ = &function->chunk;
  currentIp_ = 0;
}

}  // namespace milo::vm
