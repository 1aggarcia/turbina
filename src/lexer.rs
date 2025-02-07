use regex::Regex;

use crate::{errors::IntepreterError, models::{
    BinaryOp, Literal, Token, Type, UnaryOp
}};

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
pub fn tokenize(line: &str) -> Result<Vec<Token>, Vec<IntepreterError>> {
    // removing comments will remove a trailing newline,
    // so we have to check for it first
    let newline_regex = Regex::new("[\r\n]+$").unwrap();
    let ends_with_newline = newline_regex.is_match(line);

    let line_without_comments = line.trim().split("//").next().unwrap_or("");
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<newline>([\r\n]\s*)+)
        | (?P<fmt>[:;(),\[\]]|->)
        | (?P<binary_op>==|!=|[+*\-/%])
        | (?P<unary_op>[=!])
        | (?P<bool>true|false)
        | (?P<symbol>[a-zA-Z]\w*)
        | \d+[a-zA-Z]+ # capture illegal tokens so that remaining numbers are legal
        | (?P<int>\d+)
    "#;
    let re = Regex::new(&pattern).unwrap();
    let capture_matches = re.captures_iter(line_without_comments);
    let mut tokens = Vec::<Token>::new();
    let mut errors = Vec::<IntepreterError>::new();

    for x in capture_matches {
        let token = {
            if let Some(m) = x.name("int") {
                let int_value = m.as_str().parse::<i32>().unwrap();
                Token::Literal(Literal::Int(int_value))
            } else if let Some(_) = x.name("newline") {
                Token::Newline
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
                // ok to unwrap get() with 0, guaranteed not to be None
                let payload = x.get(0).unwrap().as_str().to_string();
                errors.push(IntepreterError::UnrecognizedToken { payload });
                continue;
            }
        };
        tokens.push(token);
    }

    if ends_with_newline {
        tokens.push(Token::Newline);
    }

    if errors.is_empty() {
        return Ok(tokens);
    } else {
        return Err(errors);
    }
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
        "!" => UnaryOp::Not,
        _ => return None,
    };
    return Some(op)
}

/// Returns keyword token if the string is a keyword, ID token otherwise
fn symbol_to_token(symbol: &str) -> Token {
    match symbol {
        "let" => Token::Let,
        "if" => Token::If,
        "else" => Token::Else,
        "string" => Token::Type(Type::String),
        "int" => Token::Type(Type::Int),
        "bool" => Token::Type(Type::Bool),
        "unknown" => Token::Type(Type::Unknown),
        "null" => Token::Null,
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

    // one token
    #[rstest]
    #[case::bool("true", bool_token(true))]
    #[case::bool("false", bool_token(false))]

    #[case::null("unknown", Token::Type(Type::Unknown))]
    #[case::null("null", Token::Null)]

    #[case::empty_string("\"\"", string_token(""))]
    #[case::normal_string(r#""hola""#, string_token("hola"))]
    #[case::string_with_spaces(r#""a b c""#, string_token("a b c"))]

    #[case::symbol("let", Token::Let)]
    #[case::symbol_with_underscore(
        "multi_word_var_name", Token::Id("multi_word_var_name".into()))]

    #[case::linux_newline("\n", Token::Newline)]
    #[case::windows_newline("\r\n", Token::Newline)]
    #[case::legacy_mac_newline("\r", Token::Newline)]
    #[case::multiple_newlines("\n\n\n\n", Token::Newline)]
    #[case::mixed_newlines("\r\n\r\n\r\r\n\n", Token::Newline)]
    #[case::newlines_with_whitespace("\n    \n  \n", Token::Newline)]
    fn one_token(#[case] line: &str, #[case] expected: Token) {
        assert_eq!(tokenize(line), Ok(vec![expected]));
    }

    // multiple tokens
    #[rstest]
    #[case::repeated_not("!!!false!", &[
        unary_op_token(UnaryOp::Not),
        unary_op_token(UnaryOp::Not),
        unary_op_token(UnaryOp::Not),
        bool_token(false),
        unary_op_token(UnaryOp::Not),
    ])]

    // identifying negative numbers is a job for the parser, not the lexer
    #[case::negative_int("-9", &[
        op_token(BinaryOp::Minus),
        int_token(9),
    ])]

    #[case::operators_and_numbers("4+5", &[
        int_token(4),
        op_token(BinaryOp::Plus),
        int_token(5),
    ])]

    #[case::operators_and_numbers("56-439%4", &[
        int_token(56),
        op_token(BinaryOp::Minus),
        int_token(439),
        op_token(BinaryOp::Percent),
        int_token(4),
    ])]

    #[case::operators_with_spaces("1* 2  +   3", &[
        int_token(1),
        op_token(BinaryOp::Star),
        int_token(2),
        op_token(BinaryOp::Plus),
        int_token(3),
    ])]

    #[case::var_declaration("let x = 5;", &[
        Token::Let,
        id_token("x"),
        unary_op_token(UnaryOp::Equals),
        int_token(5),
        formatter_token(";")
    ])]

    #[case::declared_type("let x: int = 5;", &[
        Token::Let,
        id_token("x"),
        formatter_token(":"),
        type_token(Type::Int),
        unary_op_token(UnaryOp::Equals),
        int_token(5),
        formatter_token(";")
    ])]

    #[case::symbols("fn customSymbol data if else", &[
        id_token("fn"),
        id_token("customSymbol"),
        id_token("data"),
        Token::If,
        Token::Else,
    ])]

    #[case::function_call("print(x + 1)", &[
        id_token("print"),
        formatter_token("("),
        id_token("x"),
        op_token(BinaryOp::Plus),
        int_token(1),
        formatter_token(")"),
    ])]

    #[case::comment_and_newline(
        "5 // this comment should not produce tokens\n",
        &[int_token(5), Token::Newline]
    )]
    #[case::tokens_surrounding_newlines("x \n \r\n \r \n 9 \n", &[
        id_token("x"),
        Token::Newline,
        int_token(9),
        Token::Newline,
    ])]
    fn many_tokens(#[case] line: &str, #[case] expected: &[Token]) {
        assert_eq!(tokenize(line), Ok(expected.to_vec()));
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
        let expected = strings.map(|s| string_token(s)).to_vec();
        assert_eq!(tokenize(input.as_str()), Ok(expected));
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
        assert_eq!(tokenize(token), Ok(vec![op_token(op)]));
    }

    #[rstest]
    #[case("=", UnaryOp::Equals)]
    #[case("!", UnaryOp::Not)]
    fn unary_operators(#[case] token: &str, #[case] op: UnaryOp) {
        assert_eq!(tokenize(token), Ok(vec![unary_op_token(op)]));
    }

    #[rstest]
    #[case("(")]
    #[case(")")]
    #[case(";")]
    #[case(",")]
    #[case("[")]
    #[case("]")]
    #[case("->")]
    fn formatters(#[case] token: &str) {
        assert_eq!(tokenize(token), Ok(vec![formatter_token(token)]));
    }

    #[test]
    fn symbol_starting_with_numbers() {
        let errors = vec![
            IntepreterError::UnrecognizedToken { payload: "23sdf".into() },
            IntepreterError::UnrecognizedToken { payload: "5l".into() }
        ];
        assert_eq!(tokenize("23sdf 5l"), Err(errors));
    }

    #[test]
    fn comments() {
        assert_eq!(tokenize("// a comment"), Ok(vec![]));
        assert_eq!(tokenize("x * 3 // another comment"), tokenize("x * 3"));
    }
}
