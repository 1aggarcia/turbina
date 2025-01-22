use crate::models::{get_literal_type, AbstractSyntaxTree, BinaryOp, ExprNode, LetNode, Literal, Program, Term, Variable};

/// Execute the statement represented by the AST on the program passed in.
/// Syntax errors will cause a panic and should be checked with the
/// `validate` function first.
pub fn evaluate(program: &mut Program, tree: &AbstractSyntaxTree) -> Literal {
    // TODO: use Result type for runtime errors
    match tree {
        AbstractSyntaxTree::Let(node) => eval_let(program, node),
        AbstractSyntaxTree::Expr(node) => eval_expr(program, node),
    }
}

/// Bind an expression to a name
fn eval_let(program: &mut Program, node: &LetNode) -> Literal {
    if program.vars.contains_key(&node.id) {
        panic!("variable '{}' already defined", node.id);
    }
    let literal_value = eval_expr(program, &node.value);

    program.vars.insert(node.id.clone(), Variable {
        datatype: get_literal_type(&literal_value),
        value: literal_value.clone(),
    });

    return literal_value;
}

/// Reduce a sequence of terms and operators to a single literal
fn eval_expr(program: &mut Program, expr: &ExprNode) -> Literal {
    let mut result = eval_term(program, &expr.first);

    for (op, term) in &expr.rest {
        let new_arg = eval_term(program, &term);
        result = eval_binary_op(result, op, new_arg);
    }

    return result;

}

fn eval_term(program: &mut Program, term: &Term) -> Literal {
    #[inline(always)]
    fn eval_negated(program: &mut Program, inner_term: &Term) -> Literal {
        let inner_result = eval_term(program, inner_term);
        match inner_result {
            Literal::Bool(bool) => Literal::Bool(!bool),
            Literal::Int(int) => Literal::Int(-int),
            _ => panic!("expected bool or int, got {:?}", inner_result)
        }
    }

    match term {
        Term::Literal(lit) => lit.clone(),
        Term::Id(id) => eval_id(program, id),
        Term::Not(t) | Term::Minus(t) => eval_negated(program, t),
        Term::Expr(expr) => eval_expr(program, expr),
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

/// Helper to compute the result of the binary operation
fn eval_binary_op(left: Literal, operator: &BinaryOp, right: Literal) -> Literal {
    match operator {
        BinaryOp::Plus => eval_plus(left, right),
        BinaryOp::Minus =>
            Literal::Int(literal_as_int(left) - literal_as_int(right)),
        BinaryOp::Star =>
            Literal::Int(literal_as_int(left) * literal_as_int(right)),
        BinaryOp::Slash =>
            Literal::Int(literal_as_int(left) / literal_as_int(right)),
        BinaryOp::Percent =>
            Literal::Int(literal_as_int(left) % literal_as_int(right)),

        // these use the derived `PartialEq` trait on enum `Literal`
        BinaryOp::Equals => Literal::Bool(left == right),
        BinaryOp::NotEq => Literal::Bool(left != right),
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
    use crate::{lexer::tokenize, models::{test_utils::term_tree, Type, Variable}, parser::parse};

    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(Literal::Bool(true))]
    #[case(Literal::Int(5))]
    #[case(Literal::String("so many tesssssts".to_string()))]
    fn it_returns_input_on_literals(#[case] literal: Literal) {
        let input = term_tree(Term::Literal(literal.clone()));
        assert_eq!(evaluate_fresh(&input), literal);
    }

    #[rstest]
    #[case("!false", Literal::Bool(true))]
    #[case("!true", Literal::Bool(false))]
    #[case("-15", Literal::Int(-15))]
    fn it_negates_literals(#[case] input: &str, #[case] expected: Literal) {
        let tree = make_tree(input);
        assert_eq!(evaluate_fresh(&tree), expected);
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
    #[case("3 * 2 - 5", 1)]
    #[case("3 * (2 - 5)", -9)]
    #[case("(25 % (18 / 3)) - (10 + 4)", -13)]
    #[case("(25 % 18 / 3) - (10 + 4)", -12)]
    // TODO: thorough PEMDAS test
    fn it_evaluates_complex_expressions(
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
