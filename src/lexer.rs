use regex::Regex;

use crate::models::{
    BinaryOp, Literal, Token, Type, UnaryOp
};

/// Parse source code text into a list of tokens according to the language's
/// grammar. All whitespace is eliminated, unless is is part of a string.
///
/// Tokens are one of the following:
/// - int literals: 0-9 (not including negative numbers)
/// - string literals: "..."
/// - boolean literals: "true", "false"
/// - keywords/symbols: a-zA-Z
/// - operators: +, -, &&, ||, <, =, ==
/// - formatters: (parenthesis, brackets, semicolon, comma)
/// 
/// Comments are sequences starting with `//`. Comments do not produce tokens.
pub fn tokenize(line: &str) -> Vec<Token> {
    let line_without_comments = line.split("//").next().unwrap_or("");
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<binary_op>==|!=|[+*\-/%])
        | (?P<unary_op>=)
        | (?P<fmt>[:;(),\[\]])
        | (?P<bool>true|false)
        | (?P<symbol>[a-zA-Z]\w*)
        | \d+[a-zA-Z] # capture illegal tokens so that remaining numbers are legal
        | (?P<int>\d+)
    "#;
    let re = Regex::new(&pattern).unwrap();
    let capture_matches = re.captures_iter(line_without_comments);
    let tokens= capture_matches
        .map(|x| {
            if let Some(m) = x.name("int") {
                let int_value = m.as_str().parse::<i32>().unwrap();
                Token::Literal(Literal::Int(int_value))
            } else if let Some(m) = x.name("binary_op") {
                let op = string_to_binary_op(m.as_str()).unwrap();
                Token::BinaryOp(op)
            } else if let Some(m) = x.name("unary_op") {
                let op = string_to_unary_op(m.as_str()).unwrap();
                Token::UnaryOp(op)
            } else if let Some(m) = x.name("fmt") {
                Token::Formatter(m.as_str().to_string())
            } else if let Some(m) = x.name("bool") {
                let bool_value = m.as_str() == "true";
                Token::Literal(Literal::Bool(bool_value))
            } else if let Some(m) = x.name("string") {
                let string_value = unwrap_string(m.as_str());
                Token::Literal(Literal::String(string_value))
            } else if let Some(m) = x.name("symbol") {
                symbol_to_token(m.as_str())
            } else {
                panic!("Unrecognized token: {:?}", x);
            }
        })  
        .collect::<Vec<Token>>();

    return tokens;
}

fn string_to_binary_op(string: &str) -> Option<BinaryOp> {
    let op = match string {
        "+" => BinaryOp::Plus,
        "-" => BinaryOp::Minus,
        "*" => BinaryOp::Star,
        "/" => BinaryOp::Slash,
        "%" => BinaryOp::Percent,
        "==" => BinaryOp::Equals,
        "!=" => BinaryOp::NotEq,
        _ => return None,
    };
    return Some(op)
}

fn string_to_unary_op(string: &str) -> Option<UnaryOp> {
    let op = match string {
        "=" => UnaryOp::Equals,
        _ => return None,
    };
    return Some(op)
}

/// Returns keyword token if the string is a keyword, ID token otherwise
fn symbol_to_token(symbol: &str) -> Token {
    match symbol {
        "let" => Token::Let,
        "string" => Token::Type(Type::String),
        "int" => Token::Type(Type::Int),
        "bool" => Token::Type(Type::Bool),
        _ => Token::Id(symbol.to_string()),
    }
}

/// Removes the first and last character from a string slice
fn unwrap_string(string: &str) -> String {
    let mut chars = string.chars();
    chars.next();
    chars.next_back();
    return chars.as_str().to_string();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::test_utils::*;
    use rstest::rstest;

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
            op_token(BinaryOp::Minus),
            int_token(9),
        ]);
    }

    #[rstest]
    #[case("+", BinaryOp::Plus)]
    #[case("-", BinaryOp::Minus)]
    #[case("*", BinaryOp::Star)]
    #[case("/", BinaryOp::Slash)]
    #[case("%", BinaryOp::Percent)]
    #[case("==", BinaryOp::Equals)]
    #[case("!=", BinaryOp::NotEq)]
    fn binary_operators(#[case] token: &str, #[case] op: BinaryOp) {
        assert_eq!(tokenize(token), [op_token(op)]);
    }

    // this will include the ! operator in the future
    #[rstest]
    #[case("=", UnaryOp::Equals)]
    fn unary_operators(#[case] token: &str, #[case] op: UnaryOp) {
        assert_eq!(tokenize(token), [unary_op_token(op)]);
    }

    #[rstest]
    #[case("(")]
    #[case(")")]
    #[case(";")]
    #[case(",")]
    #[case("[")]
    #[case("]")]
    fn formatters(#[case] token: &str) {
        assert_eq!(tokenize(token), [formatter_token(token)]);
    }

    #[test]
    fn operators_and_numbers() {
        let expected_basic = [
            int_token(4),
            op_token(BinaryOp::Plus),
            int_token(5),
        ];
        assert_eq!(tokenize("4+5"),  expected_basic);

        let expected_long = [
            int_token(56),
            op_token(BinaryOp::Minus),
            int_token(439),
            op_token(BinaryOp::Percent),
            int_token(4),
        ];
        assert_eq!(tokenize("56-439%4"),  expected_long);
    }

    #[test]
    fn operators_with_spaces() {
        let expected = [
            int_token(1),
            op_token(BinaryOp::Star),
            int_token(2),
            op_token(BinaryOp::Plus),
            int_token(3),
        ];
        assert_eq!(tokenize("1* 2  +   3"), expected);
    }

    #[test]
    fn one_symbol() {
        assert_eq!(tokenize("let"), [Token::Let]);
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
    fn symbol_with_underscore() {
        let expected = vec![Token::Id("multi_word_var_name".to_string())];
        assert_eq!(tokenize("multi_word_var_name"), expected);
    }

    // TODO: dont panic for invalid tokens, have `tokenize` return detailed errors
    #[test]
    #[should_panic]
    fn symbol_starting_with_numbers() {
        tokenize("23sdf");
    }

    #[test]
    fn var_declaration() {
        let expected = [
            Token::Let,
            id_token("x"),
            unary_op_token(UnaryOp::Equals),
            int_token(5),
            formatter_token(";")
        ];
        assert_eq!(tokenize("let x = 5;"), expected);
    }

    #[test]
    fn var_declaration_with_type() {
        let expected = [
            Token::Let,
            id_token("x"),
            formatter_token(":"),
            type_token(Type::Int),
            unary_op_token(UnaryOp::Equals),
            int_token(5),
            formatter_token(";")
        ];
        assert_eq!(tokenize("let x: int = 5;"), expected);
    }

    #[test]
    fn function_call() {
        let expected = [
            id_token("print"),
            formatter_token("("),
            id_token("x"),
            op_token(BinaryOp::Plus),
            int_token(1),
            formatter_token(")"),
        ];
        assert_eq!(tokenize("print(x + 1)"), expected);
    }

    #[test]
    fn comments() {
        assert_eq!(tokenize("// a comment"), vec![]);
        assert_eq!(tokenize("x * 3 // another comment"), tokenize("x * 3"));
    }
}
