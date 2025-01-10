use core::fmt;
use std::io::{stdin, stdout, Write};
use regex::Regex;

#[derive(PartialEq, Debug, Clone)]
enum TokenV2 {
    Literal(Literal),
    Operator(Operator),
    Id(String),
    Formatter(String),

    // keywords
    Let,
}

#[derive(PartialEq, Debug, Clone)]
enum Literal {
    Int(i32),
    String(String),
    Bool(bool),
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equals,
}

#[derive(PartialEq)]
struct AbstractSyntaxTree {
    token: TokenV2,
    children: Vec<AbstractSyntaxTree>,
}


impl AbstractSyntaxTree {
    fn leaf(token: TokenV2) -> Self {
        Self { token, children: vec![] }
    }

    /// returns true if this node can accept another node as the next
    /// child, false otherwise
    fn can_accept(&self, node: &AbstractSyntaxTree) -> bool {
        match self.token {
            TokenV2::Operator(_) => self.children.len() < 2,
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
                "{}{:?}",
                " ".repeat(indent * 4),
                node.token
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

#[derive(Debug, PartialEq)]
enum AbstractSyntaxTreeV2 {
    Let(LetNode),
    Operator(OperatorNode),

    // leaf nodes
    Literal(Literal),
    Id(String),
}

/// For binary operators
#[derive(Debug, PartialEq)]
struct OperatorNode {
    operator: Operator,
    left: Box<AbstractSyntaxTreeV2>,
    right: Box<AbstractSyntaxTreeV2>,
}

#[derive(Debug, PartialEq)]
struct LetNode {
    id: String,
    value: Box<AbstractSyntaxTreeV2>
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
        let tokens: Vec<TokenV2> = tokenize(next_line);
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
/// - int literals: 0-9 (not including negative numbers)
/// - string literals: "..."
/// - boolean literals: "true", "false"
/// - keywords/symbols: a-zA-Z
/// - operators: +, -, &&, ||, <, =, ==
/// - formatters: (parenthesis, brackets, semicolon, comma)
fn tokenize(line: &str) -> Vec<TokenV2> {
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<op>[+*-/=%])
        | (?P<fmt>[;()])
        | (?P<int>\d+)
        | (?P<bool>true|false)
        | (?P<symbol>\w+)
    "#;
    let re = Regex::new(&pattern).unwrap();
    let capture_matches = re.captures_iter(line);
    let tokens= capture_matches
        .map(|x| {
            if let Some(m) = x.name("int") {
                let int_value = m.as_str().parse::<i32>().unwrap();
                TokenV2::Literal(Literal::Int(int_value))
            } else if let Some(m) = x.name("op") {
                let op = string_to_operator(m.as_str()).unwrap();
                TokenV2::Operator(op)
            } else if let Some(m) = x.name("fmt") {
                TokenV2::Formatter(m.as_str().to_string())
            } else if let Some(m) = x.name("bool") {
                let bool_value = m.as_str() == "true";
                TokenV2::Literal(Literal::Bool(bool_value))
            } else if let Some(m) = x.name("string") {
                let string_value = unwrap_string(m.as_str());
                TokenV2::Literal(Literal::String(string_value))
            } else if let Some(m) = x.name("symbol") {
                symbol_to_token(m.as_str())
            } else {
                panic!("unknown type: {:?}", x);
            }
        })  
        .collect::<Vec<TokenV2>>();

    return tokens;
}

fn string_to_operator(string: &str) -> Option<Operator> {
    let op = match string {
        "+" => Operator::Plus,
        "-" => Operator::Minus,
        "*" => Operator::Star,
        "/" => Operator::Slash,
        "=" => Operator::Equals,
        "%" => Operator::Percent,
        _ => return None,
    };
    return Some(op)
}

/// Returns keyword token if the string is a keyword, ID token otherwise
fn symbol_to_token(symbol: &str) -> TokenV2 {
    match symbol {
        "let" => TokenV2::Let,
        _ => TokenV2::Id(symbol.to_string()),
    }
}

/// Removes the first and last character from a string slice
fn unwrap_string(string: &str) -> String {
    let mut chars = string.chars();
    chars.next();
    chars.next_back();
    return chars.as_str().to_string();
}

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
fn parse(tokens: Vec<TokenV2>) -> Result<AbstractSyntaxTreeV2, String> {
    if tokens.is_empty() {
        return Err("Cannot parse empty tokens".to_string());
    }
    // first token determines the type of the root node
    let head = &tokens[0];
    let head_as_leaf = token_to_leaf(head);
    if tokens.len() == 1 {
        return match head_as_leaf {
            Some(leaf) => Ok(leaf),
            None => Err(format!("cannot parse single token: {:?}", head))
        };
    }

    match head {
        TokenV2::Literal(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, 1),
        TokenV2::Id(_) =>
            build_binary_node(head_as_leaf.unwrap(), &tokens, 1),
        // TokenV2::Let => build_let(tokens[1..]),
        // more keyword cases
        TokenV2::Operator(_) => Err("Cannot parse operator as first token".to_string()),
        _ => Err(format!("unsupported token: {:?}", head))
    }
}

/// Create a AST for a binary operator given the left argument and
/// the remaining tokens (indicated by `position`).
/// 
/// tokens[position] should be the first token after `left_arg`
fn build_binary_node(
    left_arg: AbstractSyntaxTreeV2,
    tokens: &Vec<TokenV2>,
    position: usize
) -> Result<AbstractSyntaxTreeV2, String> {
    let operator_token = &tokens[position];
    let operator = match operator_token {
        TokenV2::Operator(o) => o,
        _ => return Err(format!("Invalid operator: {:?}", operator_token)),
    };

    // TODO: handle infinite args, not just one
    // TODO: range check
    let right_token = &tokens[position + 1];
    let right_arg = match token_to_leaf(right_token) {
        Some(t) => t,
        None => return Err(format!("invalid right token: {:?}", right_token))
    };

    let node = OperatorNode {
        operator: *operator,
        left: Box::new(left_arg),
        right: Box::new(right_arg),
    };
    return Ok(AbstractSyntaxTreeV2::Operator(node));
}

fn token_to_leaf(token: &TokenV2) -> Option<AbstractSyntaxTreeV2> {
    match token {
        TokenV2::Id(i) => Some(AbstractSyntaxTreeV2::Id(i.clone())),
        TokenV2::Literal(l) => Some(AbstractSyntaxTreeV2::Literal(l.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use super::*;

    fn bool_token(data: bool) -> TokenV2 {
        TokenV2::Literal(Literal::Bool(data))
    }

    fn string_token(data: &str) -> TokenV2 {
        TokenV2::Literal(Literal::String(data.to_string()))
    }

    fn int_token(data: i32) -> TokenV2 {
        TokenV2::Literal(Literal::Int(data))
    }

    fn op_token(operator: Operator) -> TokenV2 {
        TokenV2::Operator(operator)
    }

    fn id_token(data: &str) -> TokenV2 {
        TokenV2::Id(data.to_string())
    }

    fn formatter_token(data: &str) -> TokenV2 {
        TokenV2::Formatter(data.to_string())
    }

    mod test_tokenize {
        use super::*;

        #[test]
        fn booleans() {
            assert_eq!(tokenize("true"), [bool_token(true)]);
            assert_eq!(tokenize("false"), [bool_token(false)]);
        }

        #[test]
        fn empty_string() {
            let expected = [string_token("")];
            assert_eq!(tokenize("\"\""), expected);
        }

        #[test]
        fn normal_string() {
            let expected = [string_token("hola")];
            assert_eq!(tokenize("\"hola\""), expected);
        }

        #[test]
        fn string_with_spaces() {
            let expected = [string_token("a b c")];
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
            ];

            let input = strings
                .map(|s| "\"".to_string() + s + "\"")
                .join(" ");
            let expected = strings.map(|s| string_token(s));
            assert_eq!(tokenize(input.as_str()), expected);
        }

        // identifying negative numbers is a job for the parser, not the lexer
        #[test]
        fn negative_int() {
            assert_eq!(tokenize("-9"), [
                op_token(Operator::Minus),
                int_token(9),
            ]);
        }

        #[rstest]
        #[case("+", Operator::Plus)]
        #[case("-", Operator::Minus)]
        #[case("*", Operator::Star)]
        #[case("/", Operator::Slash)]
        #[case("%", Operator::Percent)]
        #[case("=", Operator::Equals)]
        fn operators(#[case] token: &str, #[case] op: Operator) {
            assert_eq!(tokenize(token), [op_token(op)]);
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
                int_token(4),
                op_token(Operator::Plus),
                int_token(5),
            ];
            assert_eq!(tokenize("4+5"),  expected_basic);

            let expected_long = [
                int_token(56),
                op_token(Operator::Minus),
                int_token(439),
                op_token(Operator::Percent),
                int_token(4),
            ];
            assert_eq!(tokenize("56-439%4"),  expected_long);
        }

        #[test]
        fn operators_with_spaces() {
            let expected = [
                int_token(1),
                op_token(Operator::Star),
                int_token(2),
                op_token(Operator::Plus),
                int_token(3),
            ];
            assert_eq!(tokenize("1* 2  +   3"), expected);
        }

        #[test]
        fn one_symbol() {
            assert_eq!(tokenize("let"), [TokenV2::Let]);
        }

        #[test]
        fn many_symbols() {
            let expected = [
                id_token("fn"),
                id_token("customSymbol"),
                id_token("data"),
            ];
            assert_eq!(tokenize("fn customSymbol data"), expected);
        }

        #[test]
        fn bad_symbol() {
            assert_eq!(tokenize("23sdf"), [id_token("23sdf")]);
        }

        #[test]
        fn var_declaration() {
            let expected = [
                TokenV2::Let,
                id_token("x"),
                op_token(Operator::Equals),
                int_token(5),
                formatter_token(";")
            ];
            assert_eq!(tokenize("let x = 5;"), expected);
        }

        #[test]
        fn function_call() {
            let expected = [
                id_token("print"),
                formatter_token("("),
                id_token("x"),
                op_token(Operator::Plus),
                int_token(1),
                formatter_token(")"),
            ];
            assert_eq!(tokenize("print(x + 1)"), expected);
        }
    }

    mod test_parse {
        use super::*;

        #[test]
        fn it_returns_error_for_empty_list()  {
            assert!(matches!(
                parse(vec![]),
                Err { .. }
            ));
        }

        #[rstest]
        #[case(int_token(2), AbstractSyntaxTreeV2::Literal(Literal::Int(2)))]
        #[case(string_token("prueba test"), AbstractSyntaxTreeV2::Literal(Literal::String("prueba test".to_string())))]
        #[case(id_token("name"), AbstractSyntaxTreeV2::Id("name".to_string()))]
        fn it_parses_one_token_to_one_node(
            #[case] token: TokenV2,
            #[case] node: AbstractSyntaxTreeV2
        ) {
            let input = vec![token.clone()];
            assert_eq!(parse(input), Ok(node));
        }

        #[rstest]
        #[case(op_token(Operator::Plus))]
        #[case(op_token(Operator::Minus))]
        #[case(op_token(Operator::Equals))]
        fn it_returns_error_for_one_operator(#[case] op: TokenV2) {
            assert!(matches!(parse(vec![op]), Err { .. }));
        }

        #[rstest]
        #[case(tokenize("3 + 2"), Operator::Plus, 3, 2)]
        #[case(tokenize("1 % 4"), Operator::Percent, 1, 4)]
        #[case(tokenize("1 - 8"), Operator::Minus, 1, 8)]
        fn it_parses_binary_expressions(
            #[case] input: Vec<TokenV2>,
            #[case] operator: Operator,
            #[case] left_val: i32,
            #[case] right_val: i32,
        ) {
            let left = Box::new(
                AbstractSyntaxTreeV2::Literal(Literal::Int(left_val))
            );
            let right = Box::new(
                AbstractSyntaxTreeV2::Literal(Literal::Int(right_val))
            );
            let expected = AbstractSyntaxTreeV2::Operator(
                OperatorNode { operator, left, right }
            );
            assert_eq!(parse(input), Ok(expected));
        }

        #[test]
        fn it_returns_error_for_multiple_ints() {
            let input = tokenize("3 2");
            assert!(matches!(parse(input), Err { .. }));
        }

        #[test]
        fn it_returns_error_for_int_plus_string() {
            let input = tokenize("1 + \"string\"");
            assert!(matches!(parse(input), Err { .. }));
        }

        #[test]
        fn it_parses_string_plus_string() {
            let input = tokenize("\"a\" + \"b\"");

            let operator = Operator::Plus;
            let left = Box::new(
                AbstractSyntaxTreeV2::Literal(Literal::String("a".to_string()))
            );
            let right = Box::new(
                AbstractSyntaxTreeV2::Literal(Literal::String("b".to_string()))
            );
            let expected = AbstractSyntaxTreeV2::Operator(
                OperatorNode { operator, left, right }
            );
            assert_eq!(parse(input), Ok(expected)); 
        }

        #[test]
        fn it_parses_var_binding() {
            let input = tokenize("let x = 4;");
            let let_node = LetNode {
                id: "x".to_string(),
                value: Box::new(AbstractSyntaxTreeV2::Literal(Literal::Int(4))),
            };
            let expected = AbstractSyntaxTreeV2::Let(let_node);
            assert_eq!(parse(input), Ok(expected));
        }
    }
}
