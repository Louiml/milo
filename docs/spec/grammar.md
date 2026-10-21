---
title: The grammar
description: The normative EBNF for Milo's concrete syntax.
section: Reference
order: 2
---

# The grammar

The normative grammar for Milo's concrete syntax lives in
[`grammar.ebnf`](https://github.com/milo-lang/milo/blob/main/docs/spec/grammar.ebnf),
alongside the specification. It is a single file of ISO/IEC 14977 EBNF, and
it is the source of truth for the syntax.

The grammar is small on purpose. Milo's sentences are not free-form English:
each one is a fixed template, and the templates are listed in the
specification. That is what keeps the language unambiguous and analysable.

## The shape of it

```ebnf
program         = { statement } , EOF ;

statement       = simple_stmt | compound_stmt ;

simple_stmt     = ( let_stmt
                   | assign_stmt
                   | return_stmt
                   | break_stmt
                   | continue_stmt
                   | import_stmt
                   | expr_stmt
                   | sentence_stmt
                   ) , { ";" } , NEWLINE ;

compound_stmt   = if_stmt | while_stmt | for_stmt | fn_stmt
                 | block
                 | sentence_block ;
```

The rest of the grammar covers the expressions, the types, and the sentence
templates. It is about four hundred lines, and it is worth reading in full
before changing the language.
