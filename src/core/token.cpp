#include "core/token.hpp"

namespace milo {

const char* to_string(TokenKind kind) {
  switch (kind) {
    case TokenKind::EndOfFile:
      return "end of file";
    case TokenKind::Newline:
      return "end of line";
    case TokenKind::Indent:
      return "an indented block";
    case TokenKind::Dedent:
      return "the end of a block";
    case TokenKind::Identifier:
      return "a name";
    case TokenKind::Integer:
      return "a whole number";
    case TokenKind::Float:
      return "a decimal number";
    case TokenKind::String:
      return "a text string";
    case TokenKind::Plus:
      return "'+'";
    case TokenKind::Minus:
      return "'-'";
    case TokenKind::Star:
      return "'*'";
    case TokenKind::Slash:
      return "'/'";
    case TokenKind::SlashSlash:
      return "'//'";
    case TokenKind::Percent:
      return "'%'";
    case TokenKind::StarStar:
      return "'**'";
    case TokenKind::Equal:
      return "'='";
    case TokenKind::EqualEqual:
      return "'=='";
    case TokenKind::BangEqual:
      return "'!='";
    case TokenKind::Less:
      return "'<'";
    case TokenKind::LessEqual:
      return "'<='";
    case TokenKind::Greater:
      return "'>'";
    case TokenKind::GreaterEqual:
      return "'>='";
    case TokenKind::Bang:
      return "'!'";
    case TokenKind::Arrow:
      return "'->'";
    case TokenKind::LeftParen:
      return "'('";
    case TokenKind::RightParen:
      return "')'";
    case TokenKind::LeftBracket:
      return "'['";
    case TokenKind::RightBracket:
      return "']'";
    case TokenKind::LeftBrace:
      return "'{'";
    case TokenKind::RightBrace:
      return "'}'";
    case TokenKind::Comma:
      return "','";
    case TokenKind::Colon:
      return "':'";
    case TokenKind::Dot:
      return "'.'";
    case TokenKind::Semicolon:
      return "';'";
    case TokenKind::Question:
      return "'?'";
    case TokenKind::Unknown:
      return "an unknown character";
  }
  return "a token";
}

std::string Token::describe() const {
  switch (kind) {
    case TokenKind::Identifier:
      return "the name `" + text + "`";
    case TokenKind::Integer:
    case TokenKind::Float:
      return "the number `" + text + "`";
    case TokenKind::String:
      return "the string " + text;
    default:
      return to_string(kind);
  }
}

}  // namespace milo
