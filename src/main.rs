use core::fmt;
use std::io::{stdin, stdout, Write};
use regex::Regex;

// static OPERATORS: &[&str] = &["+", "-", "*", "/", "%"];

#[derive(Debug, PartialEq, Clone)]
struct Token {
    r#type: TokenType,
    data: String,
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    String,
    Int,
    Operator,
    Symbol,
}

#[derive(PartialEq)]
struct AbstractSyntaxTree {
    token: Token,
    children: Vec<AbstractSyntaxTree>,
}

impl AbstractSyntaxTree {
    fn leaf(token: Token) -> Self {
        Self { token, children: vec![] }
    }
}

impl fmt::Debug for AbstractSyntaxTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // recursively print out children
        fn pretty_print(
            f: &mut std::fmt::Formatter<'_>,
            node: &AbstractSyntaxTree,
            indent: usize
        ) {
            writeln!(
                f,
                "{}{:?}({})",
                " ".repeat(indent * 4),
                node.token.r#type,
                node.token.data
            ).expect("failed to write to formatter");
            for child in &node.children {
                pretty_print(f, &child, indent + 1);
            }
        }
        writeln!(f, "")?;
        pretty_print(f, self, 0);
        Ok(())
    }
}

fn main() {
    println!("Starting interpreter");

    // user must enter Ctrl+C to quit
    loop {
        // get next line from user
        print!("> ");
        match stdout().flush() {
            Err(err) =>{
                println!("{err}");
                continue;
            },
            _ => {}
        }

        let mut buf = String::new();
        match stdin().read_line(&mut buf) {
            Err(err) =>{
                println!("{err}");
                continue;
            },
            _ => {}
        }
        let next_line = buf.trim();
        
        // parse line
        let tokens: Vec<Token> = tokenize(next_line);

        if tokens.is_empty() {
            continue;
        }

        println!("{tokens:?}");
        // evaluate line
        // let result = evaluate(tokens, &mut program);
        // match result {
        //     Ok(r) => println!("{}", value_to_string(r)),
        //     Err(err) => println!("{err}")
        // }
    }
}

/// Parse source code text into a list of tokens according to the language's
/// grammar. All whitespace is eliminated, unless is is part of a string.
///
/// 
/// Tokens are one of the following:
/// - int literals: 0-9
/// - string literals: "..."
/// - keywords/symbols: a-zA-Z
/// - operators: +, -, &&, ||, <, =, ==
/// - formatters: (parenthesis, brackets, semicolon, comma)
fn tokenize(line: &str) -> Vec<Token> {
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<op>[+*-/;=%()])
        | (?P<int>\d+)
        | (?P<symbol>\w+)
    "#;
    let re = Regex::new(&pattern).unwrap();
    let capture_matches = re.captures_iter(line);
    let tokens= capture_matches
        .map(|x| {
            let (r#type, m) = if let Some(m) = x.name("int") {
                (TokenType::Int, m)
            } else if let Some(m) = x.name("op") {
                (TokenType::Operator, m)
            } else if let Some(m) = x.name("string") {
                (TokenType::String, m)
            } else if let Some(m) = x.name("symbol") {
                (TokenType::Symbol, m)
            } else {
                panic!("unknown type: {:?}", x);
            };
            return Token { r#type, data: m.as_str().to_string() }
        })  
        .collect::<Vec<Token>>();

    return tokens;
}

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent only one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
fn parse(tokens: Vec<Token>) -> Result<AbstractSyntaxTree, String> {
    if tokens.is_empty() {
        return Err("Cannot parse empty tokens".to_string());
    }
    if tokens.len() == 1 {
        let token = tokens[0].clone();
        return if token.r#type == TokenType::Operator {
            Err("Cannot parse operator as single token".to_string())
        } else {
            Ok(AbstractSyntaxTree::leaf(tokens[0].clone()))
        }
    }
    return Err("unimplemented".to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::String,
            data: data.to_string(),
        }
    }

    fn int_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::Int,
            data: data.to_string(),
        }
    }

    fn op_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::Operator,
            data: data.to_string(),
        }
    }

    fn symbol_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::Symbol,
            data: data.to_string(),
        }
    }

    mod test_tokenize {
        use super::*;

        #[test]
        fn empty_string() {
            let expected = [string_token("\"\"")];
            assert_eq!(tokenize("\"\""), expected);
        }

        #[test]
        fn normal_string() {
            let expected = [string_token("\"hola\"")];
            assert_eq!(tokenize("\"hola\""), expected);
        }

        #[test]
        fn string_with_spaces() {
            let expected = [string_token("\"a b c\"")];
            assert_eq!(tokenize("\"a b c\""), expected);
        }

        #[test]
        fn multiple_strings() {
            let strings= [
                "hola mundo",
                "23~/.`=--`.1",
                "",
                "",
                "ya es hora"
            ].map(|s| "\"".to_string() + s + "\"");

            let input = strings.join(" ");
            let expected = strings.map(|s| string_token(s.as_str()));
            assert_eq!(tokenize(input.as_str()), expected);
        }

        #[test]
        fn operators() {
            assert_eq!(tokenize("+"), [op_token("+")]);
            assert_eq!(tokenize("-"), [op_token("-")]);
            assert_eq!(tokenize("*"), [op_token("*")]);
            assert_eq!(tokenize("/"), [op_token("/")]);
            assert_eq!(tokenize("%"), [op_token("%")]);
            assert_eq!(tokenize(";"), [op_token(";")]);
            assert_eq!(tokenize("="), [op_token("=")]);
            assert_eq!(tokenize("("), [op_token("(")]);
            assert_eq!(tokenize(")"), [op_token(")")]);
        }

        #[test]
        fn operators_and_numbers() {
            let expected_basic = [
                int_token("4"),
                op_token("+"),
                int_token("5"),
            ];
            assert_eq!(tokenize("4+5"),  expected_basic);

            let expected_long = [
                int_token("56"),
                op_token("-"),
                int_token("439"),
                op_token("%"),
                int_token("4"),
            ];
            assert_eq!(tokenize("56-439%4"),  expected_long);
        }

        #[test]
        fn operators_with_spaces() {
            let expected = [
                int_token("1"),
                op_token("*"),
                int_token("2"),
                op_token("+"),
                int_token("3"),
            ];
            assert_eq!(tokenize("1* 2  +   3"), expected);
        }

        #[test]
        fn one_symbol() {
            assert_eq!(tokenize("let"), [symbol_token("let")]);
        }

        #[test]
        fn many_symbols() {
            let expected = [
                symbol_token("fn"),
                symbol_token("customSymbol"),
                symbol_token("data"),
            ];
            assert_eq!(tokenize("fn customSymbol data"), expected);
        }

        #[test]
        fn bad_symbol() {
            assert_eq!(tokenize("23sdf"), [symbol_token("23sdf")]);
        }

        #[test]
        fn var_declaration() {
            let expected = [
                symbol_token("let"),
                symbol_token("x"),
                op_token("="),
                int_token("5"),
                op_token(";")
            ];
            assert_eq!(tokenize("let x = 5;"), expected);
        }

        #[test]
        fn function_call() {
            let expected = [
                symbol_token("print"),
                op_token("("),
                symbol_token("x"),
                op_token("+"),
                int_token("1"),
                op_token(")"),
            ];
            assert_eq!(tokenize("print(x + 1)"), expected);
        }
    }

    mod test_parse {
        use rstest::rstest;
        use super::*;

        fn leaf(token: Token) -> AbstractSyntaxTree {
            AbstractSyntaxTree::leaf(token)
        }

        #[test]
        fn it_returns_error_for_empty_list()  {
            assert!(matches!(
                parse(vec![]),
                Err { .. }
            ));
        }

        #[rstest]
        #[case(int_token("2"))]
        #[case(string_token("\"prueba test\""))]
        #[case(symbol_token("fn"))]
        fn it_parses_one_token_to_one_node(#[case] token: Token) {
            let input = vec![token.clone()];
            let expected = leaf(token);
            assert_eq!(parse(input), Ok(expected));
        }

        #[rstest]
        #[case(op_token("+"))]
        #[case(op_token("-"))]
        #[case(op_token("="))]
        fn it_returns_error_for_one_operator(#[case] op: Token) {
            assert!(matches!(parse(vec![op]), Err { .. }));
        }

        #[test]
        fn it_parses_binary_expression() {
            let input = vec![int_token("3"), op_token("+"), int_token("2")];
            let expected = AbstractSyntaxTree {
                token: op_token("+"),
                children: vec![
                    leaf(int_token("3")),
                    leaf(int_token("2")),
                ]
            };
            assert_eq!(parse(input), Ok(expected));
        }
    }
}
