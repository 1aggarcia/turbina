use std::collections::HashMap;

use crate::errors::{error, IntepreterError, MultiResult};
use crate::models::{
    AbstractSyntaxTree, BinaryExpr, BinaryOp, CondExpr, Expr, Function, FuncBody, FuncCall, LetNode, Literal, Program, Term, Type
};

type ValidationResult = MultiResult<TreeType>;
type SubResult = MultiResult<Type>;

#[derive(PartialEq, Debug)]
pub struct TreeType {
    pub datatype: Type,

    /// if present, datatype should be bound to this name in the global scope
    pub name_to_bind: Option<String>,
}

/// The types associated to all bindings in a typing scope.
/// Gives access to parent scope with the `parent` field
struct TypeContext<'a> {
    /// bindings created with with the `let` keyword
    variable_types: &'a HashMap<String, Type>,

    /// bindings created as function parameters
    parameter_types: &'a HashMap<String, Type>,

    /// The name being bound in a statement, if any. This allows recursive
    /// functions to reference themselves in their own definition.
    name_to_bind: Option<String>,

    parent: Option<&'a TypeContext<'a>>
}

impl TypeContext<'_> {
    /// Find the type associated to an ID, if any, in the local scope and
    /// all parent scopes.
    pub fn lookup(&self, id: &str) -> Option<Type> {
        if let Some(t) = self.variable_types.get(id) {
            debug_assert!(!self.parameter_types.contains_key(id),
                "binding should not be defined twice in the same scope");
            return Some(t.clone());
        }
        if let Some(t) = self.parameter_types.get(id) {
            return Some(t.clone());
        }
        self.parent.and_then(|parent_context| parent_context.lookup(id))
    }

    /// Recursively search through all type contexts and determine if the
    /// given ID is associated to a parameter
    pub fn contains_parameter(&self, id: &str) -> bool {
        if self.parameter_types.contains_key(id) {
            return true;
        }
        self.parent
            .map(|parent_context| parent_context.contains_parameter(id))
            .unwrap_or(false)  // base case
    }
}

/// Find syntax errors not caught while parsing
/// - Check that all symbols exist in the program
/// - Type-check all nodes
/// 
/// Return the type of the tree, and optionally a name to bind it to
pub fn validate(
    program: &Program, tree: &AbstractSyntaxTree
) -> ValidationResult {
    let mut global_context = TypeContext {
        variable_types: &program.type_context,
        parameter_types: &HashMap::new(),
        name_to_bind: None,
        parent: None,
    };

    match tree {
        AbstractSyntaxTree::Let(node) => {
            global_context.name_to_bind = Some(node.id.clone());
            validate_let(&global_context, node)
                .map(|datatype| TreeType {
                    datatype,
                    name_to_bind: Some(node.id.clone())
                })
        },
        AbstractSyntaxTree::Expr(node) => validate_expr(&global_context, node)
            .map(|datatype| TreeType { datatype, name_to_bind: None })
    }
}

fn validate_expr(context: &TypeContext, expr: &Expr) -> SubResult {
    match expr {
        Expr::Binary(b) => validate_binary_expr(context, b),
        Expr::Cond(c) => validate_cond_expr(context, c),
        Expr::Function(f) => validate_function(context, f),
    }
}

/// Check that the types for every term in the expression are valid
fn validate_binary_expr(context: &TypeContext, expr: &BinaryExpr) -> SubResult {
    let mut errors = Vec::<IntepreterError>::new();
    let mut result = None;

    match validate_term(context, &expr.first) {
        Ok(t) => result = Some(t),
        Err(e) => errors.extend(e),
    }

    for (op, term) in &expr.rest {
        let new_type = match validate_term(context, &term) {
            Ok(t) => t,
            Err(e) => {
                errors.extend(e);
                continue;
            }
        };
        let result_type = match result {
            Some(ref t) => t.clone(),
            None => continue,
        }; 

        match binary_op_return_type(result_type, *op, new_type) {
            Ok(t) => result = Some(t),
            Err(e) => errors.extend(e),
        }
    }

    if errors.is_empty() {
        return Ok(result.unwrap());
    } else {
        return Err(errors);
    }
}

/// Check that the condition is a boolean type, and the "if" and "else" branches
/// are of the same type
fn validate_cond_expr(context: &TypeContext, expr: &CondExpr) -> SubResult {
    let mut errors = Vec::<IntepreterError>::new();

    let cond_type = validate_expr(context, &expr.cond)?;
    if cond_type != Type::Bool {
        errors.push(IntepreterError::InvalidType { datatype: cond_type });
    }

    let true_type = validate_expr(context, &expr.if_true)?;
    let false_type = validate_expr(context, &expr.if_false)?;
    if true_type != false_type {
        let err = IntepreterError::MismatchedTypes {
            type1: true_type.clone(),
            type2: false_type
        };
        errors.push(err);
    }

    if errors.is_empty() {
        return Ok(true_type);
    } else {
        return Err(errors);
    }
}

/// Check that the function being called is defined and the input types match
/// the argument list
fn validate_func_call(context: &TypeContext, call: &FuncCall) -> SubResult {
    let (param_types, output_type) = match validate_term(context, &call.func)? {
        Type::Func { input, output } => (input, output),
        _ => {
            let err = IntepreterError::not_a_function(&call.func); 
            return Err(vec![err]);
        }
    };
    if call.args.len() != param_types.len() {
        let err = IntepreterError::ArgCount {
            got: call.args.len(),
            expected: param_types.len()
        };
        return Err(vec![err]);
    }
    
    let mut errors = vec![];
    for (arg, param_type) in call.args.iter().zip(param_types.iter()) {
        let arg_type = validate_expr(context, arg)?;
        if arg_type.is_assignable_to(param_type) {
            continue;
        }
        errors.push(IntepreterError::UnexpectedType {
            got: arg_type,
            expected: param_type.clone()
        });
    }

    if errors.is_empty() { Ok(*output_type) } else { Err(errors) }
}

/// For literals, check that the type matches any unary operators.
/// 
/// For non-literals, check that they exist and their type matches any
/// unary operators applied (! and -).
fn validate_term(context: &TypeContext, term: &Term) -> SubResult {
    match term {
        Term::Literal(lit) => get_literal_type(lit),
        Term::Id(id) => validate_id(context, &id),
        Term::Not(term) => validate_negated_bool(context, term),
        Term::Minus(term) => validated_negated_int(context, term),
        Term::Expr(expr) => validate_expr(context, expr),
        Term::FuncCall(call) => validate_func_call(context, call),
        Term::NotNull(term) => {
            let datatype = validate_term(context, term)?;
            if let Type::Nullable(inner_type) = datatype {
                return Ok(*inner_type);
            }
            Ok(datatype)
        }
    }
}

/// Should not be called with closures, closures are only created at runtime.
fn get_literal_type(literal: &Literal) -> SubResult {
    let datatype = match literal {
        Literal::Bool(_) => Type::Bool,
        Literal::Int(_) => Type::Int,
        Literal::String(_) => Type::String,
        Literal::Null => Type::Null,
        Literal::Closure(closure) =>
            panic!("Closure created before evaluation: {:?}", closure),
    };
    Ok(datatype)
}

/// Check that the return type can be resolved with global bindings and new
/// bindings introduced by the input parameters
fn validate_function(context: &TypeContext, function: &Function) -> SubResult {
    let mut param_types = HashMap::<String, Type>::new();

    // must also be stored as a list since maps don't have order
    let mut param_type_list = Vec::<Type>::new();
    let mut param_errors = Vec::<IntepreterError>::new();

    for (id, datatype) in function.params.clone() {
        if context.contains_parameter(&id) {
            param_errors.push(IntepreterError::ReassignError { id });
            continue;
        }
        param_types.insert(id.clone(), datatype.clone());
        param_type_list.push(datatype);
    }

    if !param_errors.is_empty() {
        return Err(param_errors);
    }

    let mut func_context = TypeContext {
        variable_types: &mut HashMap::new(),
        parameter_types: &param_types,
        name_to_bind: None,
        parent: Some(&context),
    };

    if let Some(declared_return_type) = &function.return_type {
        let func_type = Type::Func {
            input: param_type_list,
            output: Box::new(declared_return_type.clone())
        };
        let FuncBody::Expr(func_body) = &function.body else {
            return Ok(func_type);
        };

        // add recursive function to type context so it can be used in the body
        let mut variable_types = HashMap::new();
        if let Some(recursive_func_name) = &context.name_to_bind {
            variable_types.insert(
                recursive_func_name.clone(),
                func_type.clone()
            );
            func_context.variable_types = &variable_types;
        }

        let body_type = validate_expr(&func_context, &func_body)?;
        if !body_type.is_assignable_to(&declared_return_type) {
            return Err(IntepreterError
                ::bad_return_type(declared_return_type, &body_type).into());
        }
        Ok(func_type)
    } else {
        let FuncBody::Expr(func_body) = &function.body else {
            // this should never happen
            let message = "Function type is undecidable".to_string();
            return Err(IntepreterError::TypeError { message }.into());
        };

        let func_type = Type::Func {
            input: param_type_list,
            output: Box::new(validate_expr(&func_context, &func_body)?)
        };
        Ok(func_type)
    }
}

/// Check that the id exists in the program's type enviornment
/// The id may not have an associated value until evaluation
fn validate_id(context: &TypeContext, id: &String) -> SubResult {
    context.lookup(id).ok_or(error::undefined_id(id).into())
}

/// Check that the passed in term is a boolean
fn validate_negated_bool(context: &TypeContext, inner_term: &Term) -> SubResult {
    let datatype = validate_term(context, inner_term)?;
    match datatype {
        Type::Bool => Ok(datatype),
        _ => Err(vec![error::unary_op_type("!", datatype)])
    }
}

/// Check that the passed in term is an int
fn validated_negated_int(context: &TypeContext, inner_term: &Term) -> SubResult {
    let datatype = validate_term(context, inner_term)?;
    match datatype {
        Type::Int=> Ok(datatype),
        _ => Err(vec![error::unary_op_type("-", datatype)])
    }
}


/// Get the return type of a binary operator if the left and right operand
/// types are valid, otherwise return a validation error
fn binary_op_return_type(left: Type, operator: BinaryOp, right: Type) -> SubResult {
    use BinaryOp::*;

    let type_error =
        Err(error::binary_op_types(operator, &left, &right).into());

    match operator {
        // equality operators
        NotEq | Equals =>
            if left.is_assignable_to(&right) || right.is_assignable_to(&left)
                { Ok(Type::Bool) } else { type_error },

        // number comparison
        GreaterThan | GreaterThanOrEqual | LessThan | LessThanOrEqual =>
            if left == right && left == Type::Int
                { Ok(Type::Bool) } else { type_error },

        // math operators
        Plus => match (&left, right) {
            (Type::String, Type::String)
            | (Type::Int, Type::Int) => Ok(left),
            _ => type_error
        },
        Minus | Percent | Slash | Star =>
            if left == Type::Int { Ok(Type::Int) } else { type_error },
    }
}

/// Check that the expression type does not conflict with the declared type
/// and that the variable name is unique
fn validate_let(context: &TypeContext, node: &LetNode) -> SubResult {
    if let Some(_) = context.lookup(&node.id) {
        return Err(vec![error::already_defined(&node.id)]);
    }

    let expr_type = validate_expr(context, &node.value)?;
    let declared_type = match node.datatype.clone() {
        Some(t) => t,
        None => return Ok(expr_type),
    };

    if !expr_type.is_assignable_to(&declared_type) {
        let err = IntepreterError::UnexpectedType {
            got: expr_type,
            expected: declared_type
        };
        return Err(vec![err]);
    }
    // declared takes precedence if present
    return Ok(declared_type);
}

#[cfg(test)]
mod test_validate {
    use rstest::rstest;
    use crate::models::test_utils::term_tree;
    use crate::parser::test_utils::make_tree;
    use crate::{models::*, validator::*};

    mod term {
        use super::*;

        #[rstest]
        #[case(Literal::Int(3), Type::Int)]
        #[case(Literal::String("asdf".to_string()), Type::String)]
        #[case(Literal::Bool(false), Type::Bool)]
        fn returns_ok_for_literals(#[case] literal: Literal, #[case] expected: Type) {
            let tree = term_tree(Term::Literal(literal.clone()));
            assert_eq!(validate_fresh(tree), ok_without_binding(expected));
        }

        #[test]
        fn it_returns_ok_for_valid_symbol() {
            let tree = make_tree("x;");
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".to_string(), Type::Int);

            assert_eq!(validate(&program, &tree), ok_without_binding(Type::Int));
        }

        #[rstest]
        #[case(Type::String, Type::String)]
        #[case(Type::Int.to_nullable(), Type::Int)]
        fn it_performs_non_null_assertion(
            #[case] symbol_type: Type,
            #[case] casted_type: Type
        ) {
            let tree = make_tree("x!;");
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".to_string(), symbol_type);

            assert_eq!(validate(&program, &tree), ok_without_binding(casted_type));
        }

        #[test]
        fn it_returns_error_for_non_existent_symbol() {
            let tree = make_tree("x;");
            let expected = vec![error::undefined_id("x")];
            assert_eq!(validate_fresh(tree), Err(expected));
        }

        #[rstest]
        #[case("!3;", error::unary_op_type("!", Type::Int))]
        #[case("!\"str\";", error::unary_op_type("!", Type::String))]
        #[case("-false;", error::unary_op_type("-", Type::Bool))]
        #[case("-\"str\";", error::unary_op_type("-", Type::String))]
        fn it_returns_error_for_bad_negated_types(
            #[case] input: &str,
            #[case] error: IntepreterError,
        ) {
            let tree = make_tree(input);
            assert_eq!(validate_fresh(tree), Err(vec![error]));
        }
    }

    mod binary_expr {
        use super::*;

        #[rstest]
        #[case("3 + \"\";", BinaryOp::Plus, Type::Int, Type::String)]
        #[case("null + 5;", BinaryOp::Plus, Type::Null, Type::Int)]
        #[case("\"\" - \"\";", BinaryOp::Minus, Type::String, Type::String)]
        #[case("true % false;", BinaryOp::Percent, Type::Bool, Type::Bool)]
        #[case("0 == false;", BinaryOp::Equals, Type::Int, Type::Bool)]
        #[case("\"\" != 1;", BinaryOp::NotEq, Type::String, Type::Int)]

        #[case("2 >  false;", BinaryOp::GreaterThan, Type::Int, Type::Bool)]
        #[case("2 >= \"\";", BinaryOp::GreaterThanOrEqual, Type::Int, Type::String)]
        #[case("2 <  \"\";", BinaryOp::LessThan, Type::Int, Type::String)]
        #[case("2 <= false;", BinaryOp::LessThanOrEqual, Type::Int, Type::Bool)]
        fn it_returns_error_for_illegal_types(
            #[case] input: &str,
            #[case] op: BinaryOp,
            #[case] left_type: Type,
            #[case] right_type: Type
        ) {
            let tree = make_tree(input);
            let expected = error::binary_op_types(op, &left_type, &right_type);
            assert_eq!(validate_fresh(tree), Err(vec![expected]));
        }

        #[rstest]
        // right arg undefined
        #[case("a + 3;", vec![error::undefined_id("a")])]

        // left arg undefined
        #[case("1 + c;", vec![error::undefined_id("c")])]

        // both args undefined
        #[case("x + y - z;", ["x", "y", "z"].map(error::undefined_id).to_vec())]
        fn it_returns_error_for_child_error(
            #[case] input: &str, #[case] errors: Vec<IntepreterError>
        ) {
            // symbol does not exist
            let tree = make_tree(input);
            assert_eq!(validate_fresh(tree), Err(errors));
        }

        // TODO: refactor below tests into here
        #[rstest]
        #[case("2 > 2;", Type::Bool)]
        #[case("2 >= 2;", Type::Bool)]
        #[case("2 < 2;", Type::Bool)]
        #[case("2 <= 2;", Type::Bool)]
        fn it_returns_ok_for_good_operands(
            #[case] input: &str,
            #[case] evaluated_type: Type
        ) {
            let tree = make_tree(input);
            assert_eq!(validate_fresh(tree), ok_without_binding(evaluated_type))
        }


        #[test]
        fn it_returns_ok_for_int_addition() {
            let tree = make_tree("2 + 2;");
            assert_eq!(validate_fresh(tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_int_division() {
            let tree = make_tree("2 / 2;");
            assert_eq!(validate_fresh(tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_string_concatenation() {
            let tree = make_tree("\"a\" + \"b\";");
            assert_eq!(validate_fresh(tree), ok_without_binding(Type::String));
        }

        #[rstest]
        #[case(make_tree("0 == 1;"))]
        #[case(make_tree("true != false;"))]
        #[case(make_tree("\"a\" == \"b\";"))]
        fn it_returns_ok_for_boolean_operator_on_same_type(#[case] tree: AbstractSyntaxTree) {
            let expected = ok_without_binding(Type::Bool);
            assert_eq!(validate_fresh(tree), expected);
        }

        #[rstest]
        #[case(Type::Unknown, Type::Int)]
        #[case(Type::String, Type::String.to_nullable())]
        #[case(Type::Null, Type::String.to_nullable())]
        fn it_returns_ok_for_equality_of_comparable_types(
            #[case] type_a: Type, #[case] type_b: Type
        ) {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".into(), type_a);
            program.type_context.insert("b".into(), type_b);

            let expected = ok_without_binding(Type::Bool);

            // should be transitive
            assert_eq!(validate(&program, &make_tree("a == b")), expected);
            assert_eq!(validate(&program, &make_tree("a != b")), expected);
            assert_eq!(validate(&program, &make_tree("b == a")), expected);
            assert_eq!(validate(&program, &make_tree("b != a")), expected);
        }

        #[rstest]
        #[case(Type::Int, Type::String)]
        #[case(Type::Int.to_nullable(), Type::String.to_nullable())]
        fn it_returns_error_for_equality_of_disjoint_types(
            #[case] type_a: Type, #[case] type_b: Type
        ) {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".into(), type_a.clone());
            program.type_context.insert("b".into(), type_b.clone());

            let expected_eq = Err(
                vec![error::binary_op_types(BinaryOp::Equals, &type_a, &type_b)]
            );
            let expected_not_eq = Err(
                vec![error::binary_op_types(BinaryOp::NotEq, &type_a, &type_b)]
            );
            assert_eq!(validate(&program, &make_tree("a == b")), expected_eq);
            assert_eq!(validate(&program, &make_tree("a != b")), expected_not_eq);
        }
    }

    mod cond_expr {
        use super::*;

        #[test]
        fn it_returns_error_for_non_bool_condition() {
            let input = make_tree("if (3) false else true;");
            let expected = IntepreterError::InvalidType { datatype: Type::Int };

            assert_eq!(validate_fresh(input), Err(vec![expected]));
        }

        #[test]
        fn it_returns_error_for_mismatched_types() {
            let input = make_tree("if (1) 2 else \"\";");

            let expected = vec![
                IntepreterError::InvalidType { datatype: Type::Int },
                IntepreterError::MismatchedTypes {
                    type1: Type::Int,
                    type2: Type::String
                },
            ];
            assert_eq!(validate_fresh(input), Err(expected));
        }

        #[test]
        fn it_returns_ok_for_valid_types() {
            let input = make_tree("if (true) 3 else 4;");
            assert_eq!(validate_fresh(input), ok_without_binding(Type::Int));
        }
    }

    mod func_call {
        use super::*;

        #[test]
        fn it_returns_error_for_undefined_function() {
            let input = make_tree("test(5);");
            let expected = vec![
                IntepreterError::UndefinedError { id: "test".into() }
            ];
            assert_eq!(validate_fresh(input), Err(expected));
        }

        #[test]
        fn it_returns_error_for_non_function_id() {
            let tree = make_tree("five();");
            
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("five".into(), Type::Int);
            
            let err = IntepreterError::not_a_function(&Term::Id("five".into()));
            assert_eq!(validate(&program, &tree), Err(vec![err]));
        }

        #[test]
        fn it_returns_error_for_wrong_num_args() {
            let tree = make_tree("f(1, 2, 3);");
            let program = make_program_with_func("f", vec![Type::Int]);

            let err = IntepreterError::ArgCount { got: 3, expected: 1 };
            assert_eq!(validate(&program, &tree), Err(vec![err])); 
        }

        #[test]
        fn it_returns_error_for_mismatched_types() {
            let tree = make_tree(r#"f(false, "");"#);
            let program =
                make_program_with_func("f", vec![Type::Int, Type::Bool]);

            let errs = vec![
                IntepreterError::UnexpectedType { got: Type::Bool, expected: Type::Int },
                IntepreterError::UnexpectedType { got: Type::String, expected: Type::Bool },
            ];
            assert_eq!(validate(&program, &tree), Err(errs));
        }

        #[test]
        fn it_returns_ok_for_empty_defined_function() {
            let tree = make_tree("randInt();");
            let program = make_program_with_func("randInt",  vec![]);
            assert_eq!(validate(&program, &tree), ok_without_binding(Type::Int));
        }

        #[test]
        fn it_returns_ok_for_multi_arg_function() {
            let tree = make_tree("f(1, false);");
            let program = make_program_with_func("f",  vec![Type::Int, Type::Bool]);
            assert_eq!(validate(&program, &tree), ok_without_binding(Type::Int));
        }

        /// Create a `Program` with a function of name `name`,
        /// and input types `params`, and return type int in the type context
        fn make_program_with_func(name: &str, params: Vec<Type>) -> Program {
            let func_type = Type::Func { input: params, output: Box::new(Type::Int) };

            let mut program = Program::init_with_std_streams();
            program.type_context.insert(name.to_string(), func_type);
            program
        }
    }

    mod function {
        use crate::parser::test_utils::make_tree;

        use super::*;

        #[rstest]
        #[case::native_func("reverse;", &[Type::String], Type::String)]
        #[case::explicit_return_type("(): bool -> true;", &[], Type::Bool)]
        #[case::param_used_in_body("(x: int) -> x * x;", &[Type::Int], Type::Int)]
        #[case::returns_unknown("(): unknown -> 2025", &[], Type::Unknown)]
        #[case::curried_function(
            // this function returns another function
            "(x: int) -> (y: int) -> x + y;",
            // input
            &[Type::Int],
            // return type
            Type::Func { input: vec![Type::Int], output: Box::new(Type::Int) }
        )]
        #[case::higher_order_func(
            "(f: (string -> int), x: string) -> f(x) + 3",
            &[Type::func(&[Type::String], Type::Int), Type::String],
            Type::Int,
        )]
        fn it_returns_correct_function_type(
            #[case] input: &str,
            #[case] parameter_types: &[Type],
            #[case] return_type: Type,
        ) {
            let tree = make_tree(input);
            let expected = Type::Func {
                input: parameter_types.to_vec(),
                output: Box::new(return_type)
            };
            assert_eq!(validate_fresh(tree), ok_without_binding(expected));
        }

        #[test]
        fn it_returns_ok_for_reused_variable_name_as_parameter() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("x".into(), Type::Bool);

            let input = make_tree("(x: null) -> x;");
            let result = validate(&mut program, &input);

            assert_eq!(result, ok_without_binding(Type::Func {
                input: vec![Type::Null],
                output: Box::new(Type::Null),
            }));
        }

        #[rstest]
        #[case::undefined_symbol("(y: int) -> x;", &[error::undefined_id("x")])]
        #[case::bad_return_type(
            "(): bool -> \"\";",
            &[IntepreterError::bad_return_type(&Type::Bool, &Type::String)]
        )]
        #[case::reused_parameter(
            "(x: int) -> (y: bool) -> (x: int) -> x;",
            &[IntepreterError::ReassignError { id: "x".into() }]
        )]
        #[case::many_reused_parameters(
            "(x: int, y: int) -> (y: string, a: int, x: null) -> null;",
            &[
                IntepreterError::ReassignError { id: "y".into() },
                IntepreterError::ReassignError { id: "x".into() }
            ]
        )]
        #[case::recursive_function_without_return_type(
            "let f(x: int) -> f(x - 1);",
            &[IntepreterError::UndefinedError { id: "f".into() }]
        )]
        fn it_returns_error(#[case] input: &str, #[case] errors: &[IntepreterError]) {
            let tree = make_tree(input);
            assert_eq!(validate_fresh(tree), Err(errors.to_vec()));
        }

        #[test]
        fn it_returns_ok_for_recursive_function() {
            let input = make_tree("let f(x: int): int -> f(x - 1);");
            let func_type = Type::Func {
                input: vec![Type::Int],
                output: Box::new(Type::Int)
            };
            assert_eq!(validate_fresh(input), ok_with_binding("f", func_type));
        }

    }

    mod bindings {
        use crate::parser::test_utils::make_tree;

        use super::*;

        #[rstest]
        #[case::literal_with_declared_type("let x: int = 3;", "x", Type::Int)]
        #[case::math_expr("let something = 5 + 2;", "something", Type::Int)]
        #[case::string_expr(
            "let something = \"a\" + \"b\";", "something", Type::String)]

        #[case::int_as_unknown(
            "let x: unknown = 5", "x", Type::Unknown)]
        #[case::string_as_unknown(
            "let y: unknown = \"\"", "y", Type::Unknown)]
        #[case::function_as_unknown(
            "let f: unknown = () -> 3", "f", Type::Unknown)]

        #[case::int_as_nullable_type(
            "let n: int? = 3;", "n", Type::Int.to_nullable())]
        #[case::null_as_nullable_type(
                "let n: int? = null;", "n", Type::Int.to_nullable())]
        #[case::func_with_explicit_type(
            "let f: (int -> unknown) = (x: unknown): int -> 0;",
            "f",
            Type::func(&[Type::Int], Type::Unknown)
        )]
        fn it_returns_correct_type(
            #[case] input: &str,
            #[case] symbol: &str,
            #[case] datatype: Type,
        ) {
            let tree = make_tree(input);
            assert_eq!(validate_fresh(tree), ok_with_binding(symbol, datatype));
        }

        #[test]
        fn it_allows_casted_nullable_value_to_be_assigned_as_not_null() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("nullString".into(), Type::String.to_nullable());
    
            let input = make_tree("let validString: string = nullString!;");
            let expected = ok_with_binding("validString", Type::String);
            assert_eq!(validate(&mut program, &input), expected);
        }

        #[rstest]
        #[case("let x: int = \"string\";", Type::Int, Type::String)]
        #[case(
            "let f: (unknown -> int) = (x: int): unknown -> null;",
            Type::func(&[Type::Unknown], Type::Int),
            Type::func(&[Type::Int], Type::Unknown),
        )]
        fn it_returns_type_error_for_conflicting_types(
            #[case] input: &str,
            #[case] declared: Type,
            #[case] actual: Type,
        ) {
            let tree = make_tree(input);
            let error = IntepreterError::UnexpectedType {
                got: actual,
                expected: declared,
            };
            assert_eq!(validate_fresh(tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_error_for_assigning_unknown_to_int() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("a".to_string(), Type::Unknown);

            let tree = make_tree("let b: int = a;");
            let error = IntepreterError::UnexpectedType {
                got: Type::Unknown,
                expected: Type::Int
            };
            assert_eq!(validate(&program, &tree), Err(vec![error]));
        }

        #[test]
        fn it_propagates_error_in_expression() {
            let tree = make_tree("let y: string = undefined;");
            let error = error::undefined_id("undefined");
            assert_eq!(validate_fresh(tree), Err(vec![error]));
        }

        #[test]
        fn it_returns_err_for_duplicate_id() {
            let mut program = Program::init_with_std_streams();
            program.type_context.insert("b".to_string(), Type::Bool);
            let tree = make_tree("let b = true;");
            let error = error::already_defined("b");
            assert_eq!(validate(&program, &tree), Err(vec![error]));
        }
    }

    fn validate_fresh(input: AbstractSyntaxTree) -> ValidationResult {
        validate(&Program::init_with_std_streams(), &input)
    }

    fn ok_without_binding(datatype: Type) -> ValidationResult {
        Ok(TreeType { datatype, name_to_bind: None })
    }

    fn ok_with_binding(id: &str, datatype: Type) -> ValidationResult {
        Ok(TreeType { datatype, name_to_bind: Some(id.into()) })
    }
}
