use std::collections::HashMap;

use crate::models::{AbstractSyntaxTree, Program};
use crate::type_resolver::expr_type_resolver::resolve_expr_type;
use crate::type_resolver::import_type_resolver::resolve_import_type;
use crate::type_resolver::let_type_resolver::resolve_let_type;
use crate::type_resolver::shared::{TreeType, TypeContext, ValidationResult};

/// Find syntax errors not caught while parsing
/// - Check that all symbols exist in the program
/// - Type-check all nodes
/// 
/// Return the type of the tree, and optionally a name to bind it to
pub fn resolve_type(
    program: &Program, tree: &AbstractSyntaxTree
) -> ValidationResult {
    let mut global_context = TypeContext {
        variable_types: &program.type_context,
        parameter_types: &HashMap::new(),
        generic_type_parameters: &[],
        name_to_bind: None,
        parent: None,
    };
    resolve_statement_type(&mut global_context, tree)
}

pub fn resolve_statement_type(
    context: &mut TypeContext, statement: &AbstractSyntaxTree
) -> ValidationResult {
    match statement {
        AbstractSyntaxTree::Let(node) => {
            context.name_to_bind = Some(node.id.clone());
            resolve_let_type(&context, node)
                .map(|datatype| TreeType {
                    datatype,
                    name_to_bind: Some(node.id.clone())
                })
        },
        AbstractSyntaxTree::Import(import) => resolve_import_type(context, import),
        AbstractSyntaxTree::Expr(node) => resolve_expr_type(&context, node)
            .map(|datatype| TreeType { datatype, name_to_bind: None })
    }
}
