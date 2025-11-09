use crate::models::{BinaryExpr, Closure, EvalContext, Expr, FuncBody, FuncCall, Function, Literal, Term, Type};

/*
The factories for result types are major band-aids, it should not be this much
effort to define closures in library code; it's not possible to write these as
native functions since the data model will not allow Rust closures to be part
of an AST.

A "result" type is a function that accepts a handler for the success and error
variants and executes the appropriate handler:

type Result<T, E, R> = (handleSuccess: (T -> R), handleError: (E -> R)) -> R
*/


/// Create a "result" type with a list of values to pass to a user-supplied
/// handler for the success case. A "result" type is a function that
/// accepts a success and error handler and calls the appropriate handler.
pub fn create_result_from_success(
    data: Vec<Literal>,
    success_handler_param_types: &[Type],
    context: &EvalContext
) -> Literal {
    let func_call = FuncCall {
        func: Box::new(Term::Id("handleSuccess".into())),
        args: data.into_iter().map(|literal|
            Expr::Binary(BinaryExpr {
                first: Term::Literal(literal),
                rest: vec![],
            })
        ).collect(),
    };
    create_result_from_func_call(
        func_call,
        success_handler_param_types,
        context
    )
}

/// Create a "result" type with an error message to pass to a user-supplied
/// handler for the error case. A "result" type is a function that
/// accepts a success and error handler and calls the appropriate handler.
pub fn create_result_from_error(
    error: Literal,
    success_handler_param_types: &[Type],
    context: &EvalContext
) -> Literal {
    let func_call = FuncCall {
        func: Box::new(Term::Id("handleError".into())),
        args: vec![
            Expr::Binary(BinaryExpr {
                first: Term::Literal(error),
                rest: vec![],
            }),
        ],
    };
    create_result_from_func_call(
        func_call,
        success_handler_param_types,
        context
    )
}

fn create_result_from_func_call(
    func_call: FuncCall,
    success_handler_param_types: &[Type],
    context: &EvalContext
) -> Literal {
    let func_body = FuncBody::Expr(Box::new(
        Expr::Binary(BinaryExpr {
            first: Term::FuncCall(func_call), 
            rest: vec![]
        })
    ));
    let function = Function {
        type_params: vec!["T".into()],
        params: vec![
            (
                "handleSuccess".into(),
                Type::func(success_handler_param_types, generic_type("T")),
            ),
            (
                "handleError".into(),
                Type::func(&[Type::String], generic_type("T")),
            ),
        ],
        return_type: Some(generic_type("T")),
        body: func_body,
    };
    Literal::Closure(Closure {
        function,
        parent_scope: context.scope.bindings.clone()
    })
}

pub fn generic_type(type_name: &str) -> Type {
    Type::Generic(type_name.to_string())
}

pub fn generic_list(type_name: &str) -> Type {
    Type::Generic(type_name.to_string()).as_list()
}