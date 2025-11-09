use std::collections::HashMap;

use crate::errors::MultiResult;
use crate::models::Type;

pub type SubResult = MultiResult<Type>;
pub type ValidationResult = MultiResult<TreeType>;

#[derive(PartialEq, Debug)]
pub struct TreeType {
    pub datatype: Type,

    /// if present, datatype should be bound to this name in the global scope
    pub name_to_bind: Option<String>,
}

/// The types associated to all bindings in a typing scope.
/// Gives access to parent scope with the `parent` field
pub struct TypeContext<'a> {
    /// bindings created with with the `let` keyword
    pub variable_types: &'a HashMap<String, Type>,

    /// bindings created as function parameters
    pub parameter_types: &'a HashMap<String, Type>,

    /// the names of type parameters declared in function definitions
    pub generic_type_parameters: &'a [String],

    /// The name being bound in a statement, if any. This allows recursive
    /// functions to reference themselves in their own definition.
    pub name_to_bind: Option<String>,

    pub parent: Option<&'a TypeContext<'a>>
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

    /// Recursively search through all type contexts and determine if the
    /// given generic type has been declared
    pub fn contains_type_parameter(&self, type_parameter: &String) -> bool {
        if self.generic_type_parameters.contains(type_parameter) {
            return true;
        }
        self.parent
            .map(|parent_context|
                parent_context.contains_type_parameter(type_parameter))
            .unwrap_or(false)  // base case
    }
}

#[cfg(test)]
pub mod test_utils {
    use super::*;
    use crate::models::{AbstractSyntaxTree, Program, Type};
    use crate::type_resolver::resolve_type;
    use crate::type_resolver::shared::TreeType;

    pub fn resolve_type_fresh(input: AbstractSyntaxTree) -> ValidationResult {
        resolve_type(&Program::init_with_std_streams(), &input)
    }

    pub fn ok_without_binding(datatype: Type) -> ValidationResult {
        Ok(TreeType { datatype, name_to_bind: None })
    }

    pub fn ok_with_binding(id: &str, datatype: Type) -> ValidationResult {
        Ok(TreeType { datatype, name_to_bind: Some(id.into()) })
    }
}
