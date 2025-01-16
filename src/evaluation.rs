use crate::models::{get_literal_type, AbstractSyntaxTree, LetNode, Literal, Operator, OperatorNode, Program, Variable};

/// Execute the statement represented by the AST on the program passed in.
/// Syntax errors will cause a panic and should be checked with the
/// `validate` function first.
pub fn evaluate(program: &mut Program, tree: &AbstractSyntaxTree) -> Literal {
    // TODO: use Result type for runtime errors
    match tree {
        AbstractSyntaxTree::Literal(lit) => lit.clone(),
        AbstractSyntaxTree::Id(id) => eval_id(program, id),
        AbstractSyntaxTree::Let(node) => eval_let(program, node),
        AbstractSyntaxTree::Operator(node) => eval_binary_op(program, node),
    }
}

/// Lookup the value stored for a variable name
fn eval_id(program: &mut Program, id: &str) -> Literal {
    match program.vars.get(id) {
        Some(var) => var.value.clone(),

        // TODO: this should never happen, but use result type anyway
        None => panic!("variable '{}' does not exist", id),
    }
}

/// Evaluate an expression and bind it to a symbol.
/// Returns the value stored.
fn eval_let(program: &mut Program, node: &LetNode) -> Literal {
    if program.vars.contains_key(&node.id) {
        panic!("variable '{}' already defined", node.id);
    }
    let literal_value = evaluate(program, &node.value);

    program.vars.insert(node.id.clone(), Variable {
        datatype: get_literal_type(&literal_value),
        value: literal_value.clone(),
    });

    return literal_value;
}

/// Compute the result of the binary operation
fn eval_binary_op(program: &mut Program, node: &OperatorNode) -> Literal {
    let left = evaluate(program, &node.left);
    let right = evaluate(program, &node.right);

    match node.operator {
        Operator::Plus => eval_plus(left, right),
        Operator::Minus =>
            Literal::Int(literal_as_int(left) - literal_as_int(right)),
        Operator::Star =>
            Literal::Int(literal_as_int(left) * literal_as_int(right)),
        Operator::Slash =>
            Literal::Int(literal_as_int(left) / literal_as_int(right)),
        Operator::Percent =>
            Literal::Int(literal_as_int(left) % literal_as_int(right)),
        Operator::OneEq =>
            panic!("operator '=' cannot be evaluated as a binary operator"),

        // these use the derived `PartialEq` trait on enum `Literal`
        Operator::TwoEq => Literal::Bool(left == right),
        Operator::NotEq => Literal::Bool(left != right),
    }
}

/// perform either string concatenation or integer addition,
/// depending on the literal type
fn eval_plus(left: Literal, right: Literal) -> Literal {
    let left_as_str = literal_to_string(left.clone());
    match left_as_str {
        None => Literal::Int(
            literal_as_int(left) + literal_as_int(right)
        ),
        Some(str) => Literal::String(
            str + &literal_to_string(right).unwrap() 
        ),
    }
}

/// Casts a literal to an int
fn literal_as_int(literal: Literal) -> i32 {
    match literal {
        Literal::Int(i) => i,
        _ => panic!("expected int literal, got {:?}", literal),
    }
}

/// Tries to extract a string from a literal, if possible
fn literal_to_string(literal: Literal) -> Option<String> {
    match literal {
        Literal::String(str) => Some(str),
        _ => None,
    }
}

#[cfg(test)]
mod test_evalutate {
    use crate::{lexer::tokenize, models::{Type, Variable}, parser::parse};

    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(Literal::Bool(true))]
    #[case(Literal::Int(5))]
    #[case(Literal::String("so many tesssssts".to_string()))]
    fn it_returns_input_on_literals(#[case] literal: Literal) {
        let input = AbstractSyntaxTree::Literal(literal.clone());
        assert_eq!(evaluate_fresh(&input), literal);
    }

    #[test]
    fn it_looks_up_variables_correctly() {
        let mut program = Program::new();
        program.vars.insert("is_lang_good".to_string(), Variable {
            datatype: Type::Bool,
            value: Literal::Bool(true),
        });
        program.vars.insert("some_int".to_string(), Variable {
            datatype: Type::Int,
            value: Literal::Int(-5),
        });

        let input = make_tree("is_lang_good");
        assert_eq!(evaluate(&mut program, &input), Literal::Bool(true));
    }

    #[test]
    fn it_binds_literal_value_to_symbol() {
        let mut program = Program::new();
        let input = make_tree("let t = true");

        evaluate(&mut program, &input);
        assert_eq!(program.vars["t"], Variable {
            datatype: Type::Bool,
            value: Literal::Bool(true),
        });
    }

    #[test]
    fn it_returns_value_of_var_after_binding() {
        let input = make_tree("let t = 12345 - 98765");
        let expected = Literal::Int(12345 - 98765);
        assert_eq!(evaluate(&mut Program::new(), &input), expected);
    }

    #[rstest]
    #[case("3 - 5", 3 - 5)]
    #[case("3 + 5", 3 + 5)]
    #[case("3 * 5", 3 * 5)]
    #[case("3 / 5", 3 / 5)]
    #[case("3 % 5", 3 % 5)]
    fn it_evaluates_binary_math_operators(
        #[case] input: &str, #[case] expected_val: i32
    ) {
        let input = make_tree(input);
        let expected = Literal::Int(expected_val);
        assert_eq!(evaluate(&mut Program::new(), &input), expected);
    }

    #[rstest]
    // ints
    #[case("3 == 5", 3 == 5)]
    #[case("12 == 12", 12 == 12)]
    #[case("0 != 0", 0 != 0)]
    #[case("2 != 1", 2 != 1)]

    // strings
    #[case("\"a\" == \"a\"", true)]
    #[case("\"a\" != \"a\"", false)]
    #[case("\"abc\" == \"efg\"", false)]
    #[case("\"efg\" != \"abc\"", true)]

    // bools
    #[case("true == true", true)]
    #[case("true != true", false)]
    #[case("false == true", false)]
    #[case("true != false ", true)]
    fn it_evaluates_binary_bool_operators(
        #[case] input: &str, #[case] expected_val: bool
    ) {
        let input = make_tree(input);
        let expected = Literal::Bool(expected_val);
        assert_eq!(evaluate(&mut Program::new(), &input), expected);
    }

    #[test]
    fn it_evaluates_string_concatenation() {
        let input = make_tree("\"abc\" + \"xyz\"");
        let expected = Literal::String("abcxyz".to_string());
        assert_eq!(evaluate(&mut Program::new(), &input), expected);
    }

    // evaluate an AST on a new program
    fn evaluate_fresh(tree: &AbstractSyntaxTree) -> Literal{
        return evaluate(&mut Program::new(), &tree);
    }

    fn make_tree(statement: &str) -> AbstractSyntaxTree {
        return parse(tokenize(statement)).unwrap();
    }
}
