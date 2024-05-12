// deno-lint-ignore-file no-empty-interface
// https://github.com/tlaceby/guide-to-interpreters-series

// -----------------------------------------------------------
// --------------          AST TYPES        ------------------
// ---     Defines the structure of our languages AST      ---
// -----------------------------------------------------------

export type NodeType =
	// STATEMENTS
	| "Program"
	| "VarDeclaration"
	| "ArgumentType"
	| "FunctionDeclaration"
	| "ConditionalDeclaration"
	| "ForLoopDeclaration"
	// EXPRESSIONS
	| "AssignmentExpr"
	| "MemberExpr"
	| "CallExpr"
	// Literals
	| "Property"
	| "ObjectLiteral"
	| "NumericLiteral"
	| "Identifier"
	| "UnaryExpr"
	| "BinaryExpr"
	| "RelationalExpr"
	| "LogicalExpr"
	| "EqualityExpr"
	| "CompoundExpr"
;

/**
 * Statements do not result in a value at runtime.
 They contain one or more expressions internally */
export interface Stmt {
	kind: NodeType;
	_context?: {
		body: Stmt[],
		globals: Stmt[]
	}
}

/**
 * Defines a block which contains many statements.
 * -  Only one program will be contained in a file.
 */
export interface Program extends Stmt {
	kind: "Program";
	body: Stmt[];
}

export interface VarDeclaration extends Stmt {
	kind: "VarDeclaration";
	type: string;
	identifier: string;
	pointer: boolean;
	value?: Expr;
}

export interface FunctionDeclaration extends Stmt {
	kind: "FunctionDeclaration";
	qualifiers: string[];
	parameters: VarDeclaration[];
	name: string;
	type: string;
	body: Stmt[];
}

export interface ForLoopDeclaration extends Stmt {
	kind: "ForLoopDeclaration";
	init: Expr;
	condition: Expr;
	increment: Expr;
	body: Stmt[];
}

export interface ConditionalDeclaration extends Stmt {
	kind: "ConditionalDeclaration";
	test: Expr;
	body: Stmt[];
}

/**  Expressions will result in a value at runtime unlike Statements */
export interface Expr extends Stmt {}

export interface AssignmentExpr extends Expr {
	kind: "AssignmentExpr";
	assigne: Expr;
	value: Expr;
}

/**
 * A operation with two sides seperated by a operator.
 * Both sides can be ANY Complex Expression.
 * - Supported Operators -> + | - | / | * | %
 */
export interface BinaryExpr extends Expr {
	kind: "BinaryExpr";
	left: Expr;
	right: Expr;
	operator: string;
}

export interface RelationalExpr extends Expr {
	kind: "RelationalExpr";
	left: Expr;
	right: Expr;
	operator: string;
}

export interface LogicalExpr extends Expr {
	kind: "LogicalExpr";
	left: Expr;
	right: Expr;
	operator: string;
}

export interface EqualityExpr extends Expr {
	kind: "EqualityExpr";
	left: Expr;
	right: Expr;
	operator: string;
}

export interface CompoundExpr extends Expr {
	kind: "CompoundExpr";
	left: Expr;
	right: Expr;
	operator: string;
}

export interface UnaryExpr extends Expr {
	kind: "UnaryExpr";
	operand: Expr;
	operator: string;
}

export interface CallExpr extends Expr {
	kind: "CallExpr";
	args: Expr[];
	caller: Expr;
}

export interface MemberExpr extends Expr {
	kind: "MemberExpr";
	object: Expr;
	property: Expr;
	computed: boolean;
}

// LITERAL / PRIMARY EXPRESSION TYPES
/**
 * Represents a user-defined variable or symbol in source.
 */
export interface Identifier extends Expr {
	kind: "Identifier";
	symbol: string;
}

/**
 * Represents a numeric constant inside the soure code.
 */
export interface NumericLiteral extends Expr {
	kind: "NumericLiteral";
	value: number;
}

export interface Property extends Expr {
	kind: "Property";
	key: string;
	value?: Expr;
}

export interface ObjectLiteral extends Expr {
	kind: "ObjectLiteral";
	properties: Property[];
}