use std::ascii::escape_default;

use regex::Regex;
use unescaper::unescape;

use crate::errors::{InterpreterError, MultiResult, Result};
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
/// - formatters: (parentheses, brackets, semicolon, comma)
/// 
/// Comments are sequences starting with `//`. Comments do not produce tokens.
pub fn tokenize(line: &str) -> MultiResult<Vec<Token>> {
    // removing comments will remove a trailing newline,
    // so we have to check for it first
    let newline_regex = Regex::new("[\r\n]+$").unwrap();
    let ends_with_newline = newline_regex.is_match(line);

    let line_without_comments = line.trim().split("//").next().unwrap_or("");
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<newline>([\r\n]\s*)+)
        | (?P<fmt>[:;\.(),\[\]{}]|->)
        | (?P<binary_op>==|!=|<=|>=|&&|\|>|\|\||[+*\-/%<>])
        | (?P<unary_op>[=!?])
        | (?P<bool>true|false)
        | (?P<symbol>[a-zA-Z_]\w*)
        | \d+[a-zA-Z]+ # capture illegal tokens so that remaining numbers are legal
        | (?P<int>\d+)
    "#;
    let re = Regex::new(&pattern).unwrap();
    let capture_matches = re.captures_iter(line_without_comments);
    let mut tokens = Vec::<Token>::new();
    let mut errors = Vec::<InterpreterError>::new();

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
                formatter_to_token(m.as_str())?
            } else if let Some(m) = x.name("bool") {
                let bool_value = m.as_str() == "true";
                Token::Literal(Literal::Bool(bool_value))
            } else if let Some(m) = x.name("string") {
                let string_value = unescape_string(m.as_str())?;
                Token::Literal(Literal::String(string_value))
            } else if let Some(m) = x.name("symbol") {
                symbol_to_token(m.as_str())
            } else {
                // ok to unwrap get() with 0, guaranteed not to be None
                let payload = x.get(0).unwrap().as_str().to_string();
                errors.push(InterpreterError::UnrecognizedToken { payload });
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
        "&&" => BinaryOp::And,
        "||" => BinaryOp::Or,
        "!=" => BinaryOp::NotEq,
        "<" => BinaryOp::LessThan,
        "<=" => BinaryOp::LessThanOrEqual,
        ">" => BinaryOp::GreaterThan,
        ">=" => BinaryOp::GreaterThanOrEqual,
        "|>" => BinaryOp::Pipe,
        _ => return None,
    };
    return Some(op)
}

fn string_to_unary_op(string: &str) -> Option<UnaryOp> {
    let op = match string {
        "=" => UnaryOp::Equals,
        "!" => UnaryOp::Not,
        "?" => UnaryOp::Nullable,
        _ => return None,
    };
    return Some(op)
}

fn formatter_to_token(text: &str) -> MultiResult<Token> {
    let token = match text {
        ":" => Token::Colon,
        ";" => Token::Semicolon,
        "." => Token::Dot,
        "(" => Token::OpenParens,
        ")" => Token::CloseParens,
        "," => Token::Comma,
        "[" => Token::OpenSquareBracket,
        "]" => Token::CloseSquareBracket,
        "{" => Token::OpenCurlyBracket,
        "}" => Token::CloseCurlyBracket,
        "->" => Token::Arrow,
        _ => return Err(
            InterpreterError::UnrecognizedToken { payload: text.into() }.into()
        )
    };
    Ok(token)
}

/// Returns keyword token if the string is a keyword, ID token otherwise
fn symbol_to_token(symbol: &str) -> Token {
    match symbol {
        "let" => Token::Let,
        "import" => Token::Import,
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

/// Convert raw string data stored by the program into a displayable version
/// which shows all escape characters. Ex: `"a\nb"` becomes `"\"a\\nb\""` so
/// that it is shown to the user as `"a\nb"`.
pub fn escape_string(string: &str) -> Result<String> {
    let escaped_string_bytes = string
        .as_bytes()
        .iter()
        .flat_map(|byte| escape_default(*byte))
        .collect();

    match String::from_utf8(escaped_string_bytes) {
        Ok(escaped_string) => Ok(format!("\"{escaped_string}\"")),
        Err(err) => Err(
            InterpreterError::SyntaxError { message: err.to_string() }
        )
    }
}

/// Inverse action of `escape_string`. Converts an input string from the lexer
/// into raw string data to be stored by the program, applying all escape
/// characters. Ex: `"\"a\\nb\""` becomes `"a\nb"`
fn unescape_string(string: &str) -> Result<String> {
    let mut chars = string.chars();
    chars.next();
    chars.next_back();
    let string_without_quotes = chars.as_str();
    unescape(string_without_quotes).map_err(|err|
        InterpreterError::SyntaxError { message: err.to_string() }
    )
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
    #[case::string_with_newline(r#""a \n b""#, string_token("a \n b"))]

    #[case::symbol("let", Token::Let)]
    #[case::symbol("import", Token::Import)]
    #[case::symbol_with_underscore(
        "multi_word_var_name", id_token("multi_word_var_name"))]
    #[case::symbol_starting_with_underscore("_a", id_token("_a"))]
    #[case::underscore_symbol("_", id_token("_"))]

    #[case::linux_newline("\n", Token::Newline)]
    #[case::windows_newline("\r\n", Token::Newline)]
    #[case::legacy_mac_newline("\r", Token::Newline)]
    #[case::multiple_newlines("\n\n\n\n", Token::Newline)]
    #[case::mixed_newlines("\r\n\r\n\r\r\n\n", Token::Newline)]
    #[case::newlines_with_whitespace("\n    \n  \n", Token::Newline)]

    #[case("(", Token::OpenParens)]
    #[case(")", Token::CloseParens)]
    #[case(";", Token::Semicolon)]
    #[case(",", Token::Comma)]
    #[case(".", Token::Dot)]
    #[case("[", Token::OpenSquareBracket)]
    #[case("]", Token::CloseSquareBracket)]
    #[case("{", Token::OpenCurlyBracket)]
    #[case("}", Token::CloseCurlyBracket)]
    #[case("->", Token::Arrow)]
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
        Token::Semicolon,
    ])]

    #[case::declared_type("let x: int = 5;", &[
        Token::Let,
        id_token("x"),
        Token::Colon,
        type_token(Type::Int),
        unary_op_token(UnaryOp::Equals),
        int_token(5),
        Token::Semicolon,
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
        Token::OpenParens, 
        id_token("x"),
        op_token(BinaryOp::Plus),
        int_token(1),
        Token::CloseParens,
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
    #[case::nullable_type("int?", &[
        Token::Type(Type::Int),
        Token::UnaryOp(UnaryOp::Nullable)])
    ]
    #[case::import_statement("import root.dir.module;", &[
        Token::Import,
        id_token("root"),
        Token::Dot,
        id_token("dir"),
        Token::Dot,
        id_token("module"),
        Token::Semicolon,
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
    #[case("&&", BinaryOp::And)]
    #[case("||", BinaryOp::Or)]
    #[case("!=", BinaryOp::NotEq)]
    #[case("<", BinaryOp::LessThan)]
    #[case("<=", BinaryOp::LessThanOrEqual)]
    #[case(">", BinaryOp::GreaterThan)]
    #[case(">=", BinaryOp::GreaterThanOrEqual)]
    #[case("|>", BinaryOp::Pipe)]
    fn binary_operators(#[case] token: &str, #[case] op: BinaryOp) {
        assert_eq!(tokenize(token), Ok(vec![op_token(op)]));
    }

    #[rstest]
    #[case("=", UnaryOp::Equals)]
    #[case("!", UnaryOp::Not)]
    #[case("?", UnaryOp::Nullable)]
    fn unary_operators(#[case] token: &str, #[case] op: UnaryOp) {
        assert_eq!(tokenize(token), Ok(vec![unary_op_token(op)]));
    }

    #[test]
    fn symbol_starting_with_numbers() {
        let errors = vec![
            InterpreterError::UnrecognizedToken { payload: "23sdf".into() },
            InterpreterError::UnrecognizedToken { payload: "5l".into() }
        ];
        assert_eq!(tokenize("23sdf 5l"), Err(errors));
    }

    #[test]
    fn comments() {
        assert_eq!(tokenize("// a comment"), Ok(vec![]));
        assert_eq!(tokenize("x * 3 // another comment"), tokenize("x * 3"));
    }

    mod escape_string {
        use super::*;

        #[test]
        fn it_escapes_escape_characters_and_adds_quotes() {
            assert_eq!(escape_string("1\n\t23").unwrap(), "\"1\\n\\t23\"")
        }

        #[test]
        fn it_performs_inverse_of_unescape_string() {
            let original = "\"1\\n\\t23\"";
            let unescaped = unescape_string(original).unwrap();
           assert_eq!(escape_string(&unescaped).unwrap(), original); 
        }
    }

    mod unescape_string {
        use super::*;

        #[test]
        fn it_applies_escape_characters_and_removes_quotes() {
            assert_eq!(unescape_string("\"\\n1\"").unwrap(), "\n1");
        }

        #[test]
        fn it_applies_inverse_of_escape_string() {
            let original = "\n1";
            let escaped = escape_string(original).unwrap();
            assert_eq!(unescape_string(&escaped).unwrap(), original);
        }
    }
}
