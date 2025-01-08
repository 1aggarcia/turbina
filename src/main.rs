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
    Symbol, // variable names, keywords
    BinaryOperator, // performs some action with two pieces of data (e.g. +, -)
    Formatter, // doesn't perform any action (e.g. ;)
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

    /// returns true if this node can accept another node as the next
    /// child, false otherwise
    fn can_accept(&self, node: &AbstractSyntaxTree) -> bool {
        match self.token.r#type {
            TokenType::BinaryOperator => self.children.len() < 2,
            _ => false,
        }
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

        let syntax_tree = parse(tokens);
        match syntax_tree {
            Ok(tree) => println!("{:?}", tree),
            Err(err) => println!("Syntax Error: {}", err),
        }
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
    // TODO: add bool
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<op>[+*-/=%])
        | (?P<fmt>[;()])
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
                (TokenType::BinaryOperator, m)
            } else if let Some(m) = x.name("fmt") {
                (TokenType::Formatter, m)
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
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
fn parse(tokens: Vec<Token>) -> Result<AbstractSyntaxTree, String> {
    if tokens.is_empty() {
        return Err("Cannot parse empty tokens".to_string());
    }
    let first_token = &tokens[0];
    let mut root = AbstractSyntaxTree::leaf(first_token.clone());
    if tokens.len() == 1 && first_token.r#type == TokenType::BinaryOperator {
        return Err("Cannot parse operator as single token".to_string())
    }
    for token in tokens[1..].iter() {
        let mut new_node = AbstractSyntaxTree::leaf(token.clone());
        match new_node.token.r#type {
            TokenType::BinaryOperator => {
                // swap references since binary operators come after first arg
                let temp = root;
                root = new_node;
                new_node = temp;
            },
            _ => {}
        }
        if root.can_accept(&new_node) {
            root.children.push(new_node);
        } else {
            return Err(format!(
                "Token {} is an invalid argument for token {}",
                new_node.token.data,
                root.token.data,
            ));
        }
    }    

    return Ok(root);
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
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
            r#type: TokenType::BinaryOperator,
            data: data.to_string(),
        }
    }

    fn symbol_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::Symbol,
            data: data.to_string(),
        }
    }

    fn formatter_token(data: &str) -> Token {
        return Token {
            r#type: TokenType::Formatter,
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
        fn negative_int() {
            assert_eq!(tokenize("-9"), [int_token("-9")]);
        }

        #[rstest]
        #[case("+")]
        #[case("-")]
        #[case("*")]
        #[case("/")]
        #[case("%")]
        #[case("=")]
        fn operators(#[case] token: &str) {
            assert_eq!(tokenize(token), [op_token(token)]);
        }

        #[rstest]
        #[case("(")]
        #[case(")")]
        #[case(";")]
        fn formatters(#[case] token: &str) {
            assert_eq!(tokenize(token), [formatter_token(token)]);
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
                formatter_token(";")
            ];
            assert_eq!(tokenize("let x = 5;"), expected);
        }

        #[test]
        fn function_call() {
            let expected = [
                symbol_token("print"),
                formatter_token("("),
                symbol_token("x"),
                op_token("+"),
                int_token("1"),
                formatter_token(")"),
            ];
            assert_eq!(tokenize("print(x + 1)"), expected);
        }
    }

    mod test_parse {
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

        #[rstest]
        #[case(tokenize("3 + 2"), AbstractSyntaxTree {
            token: op_token("+"),
            children: vec![
                leaf(int_token("3")),
                leaf(int_token("2")),
            ]
        })]
        #[case(tokenize("1 % 4"), AbstractSyntaxTree {
            token: op_token("%"),
            children: vec![
                leaf(int_token("1")),
                leaf(int_token("4")),
            ]
        })]
        #[case(tokenize("1 - 8"), AbstractSyntaxTree {
            token: op_token("-"),
            children: vec![
                leaf(int_token("1")),
                leaf(int_token("8")),
            ]
        })]
        fn it_parses_binary_expressions(
            #[case] input: Vec<Token>,
            #[case] expected: AbstractSyntaxTree
        ) {
            assert_eq!(parse(input), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_multiple_ints() {
            let input = tokenize("3 2");
            assert!(matches!(parse(input), Err { .. }));
        }

        #[test]
        fn it_parses_var_binding() {
            let input = tokenize("let x = 4;");
            let expected = AbstractSyntaxTree {
                token: symbol_token("="),
                children: vec![
                    leaf(symbol_token("x")),
                    leaf(int_token("4")),
                ]
            };
            assert_eq!(parse(input), Ok(expected));
        }
    }
}
