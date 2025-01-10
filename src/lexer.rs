use regex::Regex;

use crate::models::{
    TokenV2,
    Literal,
    Operator,
};

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
pub fn tokenize(line: &str) -> Vec<TokenV2> {
    let pattern = r#"(?x)
        (?P<string>\"[^"]*\")
        | (?P<op>[+*-/=%])
        | (?P<fmt>[;()])
        | (?P<bool>true|false)
        | (?P<symbol>[a-zA-Z]\w*)
        | \d+[a-zA-Z] # capture illegal tokens so that remaining numbers are legal
        | (?P<int>\d+)
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
                panic!("Unrecognized token: {:?}", x);
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
    fn symbol_with_underscore() {
        let expected = vec![TokenV2::Id("multi_word_var_name".to_string())];
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
