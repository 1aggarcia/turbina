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
    println!("UNWRAPPING {:?}", chars);
    return chars.as_str().to_string();
}

/// Convert a sequence of tokens into an abstract syntax tree.
/// 
/// The tokens should represent exactly one statement and therefore one syntax
/// tree. Otherwise, an error is returned.
/// A single literal value is a valid statement (e.g. "3").
/// 
/// A syntax error is returned for any syntactical errors in the token sequence
fn parse(tokens: Vec<TokenV2>) -> Result<AbstractSyntaxTree, String> {
    if tokens.is_empty() {
        return Err("Cannot parse empty tokens".to_string());
    }
    let first_token = &tokens[0];
    let mut root = AbstractSyntaxTree::leaf(first_token.clone());

    let is_operator = match first_token  {
        TokenV2::Operator(_) => true,
        _ => false,
    };

    if tokens.len() == 1 && is_operator {
        return Err("Cannot parse operator as single token".to_string())
    }
    for token in tokens[1..].iter() {
        let mut new_node = AbstractSyntaxTree::leaf(token.clone());
        if let TokenV2::Operator(_) = new_node.token {
            // swap references since binary operators come after first arg
            let temp = root;
            root = new_node;
            new_node = temp;
        }
        if root.can_accept(&new_node) {
            root.children.push(new_node);
        } else {
            return Err(format!(
                "Token {:?} is an invalid argument for token {:?}",
                new_node.token,
                root.token,
            ));
        }
    }    

    return Ok(root);
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

        fn leaf(token: TokenV2) -> AbstractSyntaxTree {
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
        #[case(int_token(2))]
        #[case(string_token("\"prueba test\""))]
        #[case(id_token("fn"))]
        fn it_parses_one_token_to_one_node(#[case] token: TokenV2) {
            let input = vec![token.clone()];
            let expected = leaf(token);
            assert_eq!(parse(input), Ok(expected));
        }

        #[rstest]
        #[case(op_token(Operator::Plus))]
        #[case(op_token(Operator::Minus))]
        #[case(op_token(Operator::Equals))]
        fn it_returns_error_for_one_operator(#[case] op: TokenV2) {
            assert!(matches!(parse(vec![op]), Err { .. }));
        }

        #[rstest]
        #[case(tokenize("3 + 2"), AbstractSyntaxTree {
            token: op_token(Operator::Plus),
            children: vec![
                leaf(int_token(3)),
                leaf(int_token(3)),
            ]
        })]
        #[case(tokenize("1 % 4"), AbstractSyntaxTree {
            token: op_token(Operator::Percent),
            children: vec![
                leaf(int_token(1)),
                leaf(int_token(4)),
            ]
        })]
        #[case(tokenize("1 - 8"), AbstractSyntaxTree {
            token: op_token(Operator::Minus),
            children: vec![
                leaf(int_token(1)),
                leaf(int_token(8)),
            ]
        })]
        fn it_parses_binary_expressions(
            #[case] input: Vec<TokenV2>,
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
        fn it_returns_error_for_int_plus_string() {
            let input = tokenize("1 + \"string\"");
            assert!(matches!(parse(input), Err { .. }));
        }

        #[test]
        fn it_parses_string_plus_string() {
            let input = tokenize("\"a\" + \"b\"");
            let expected = AbstractSyntaxTree {
                token: id_token("+"),
                children: vec![
                    leaf(string_token("\"a\"")),
                    leaf(string_token("\"b\"")),
                ]
            };
            assert_eq!(parse(input), Ok(expected)); 
        }

        #[test]
        fn it_parses_var_binding() {
            let input = tokenize("let x = 4;");
            let expected = AbstractSyntaxTree {
                token: TokenV2::Let,
                children: vec![
                    leaf(id_token("x")),
                    leaf(int_token(4)),
                ]
            };
            assert_eq!(parse(input), Ok(expected));
        }
    }
}
