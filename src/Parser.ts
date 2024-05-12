// deno-lint-ignore-file no-explicit-any
import {
	AssignmentExpr,
	BinaryExpr,
	CallExpr,
	Expr,
	Identifier,
	MemberExpr,
	NumericLiteral,
	ObjectLiteral,
	Program,
	Property,
	Stmt,
	VarDeclaration,
	FunctionDeclaration,
	RelationalExpr,
	EqualityExpr,
	UnaryExpr,
	ForLoopDeclaration,
	LogicalExpr,
	ConditionalDeclaration,
	CompoundExpr,
} from "./AST";

import { Token, tokenize, TokenType } from "./Lexer";

/**
 * Frontend for producing a valid AST from sourcode
 */
export class Parser {
	private tokens: Token[] = [];

	/*
	 * Determines if the parsing is complete and the END OF FILE Is reached.
	 */
	private not_eof(): boolean {
		return this.tokens[0].type != TokenType.EOF;
	}

	/**
	 * Returns the currently available token
	 */
	private at(index: number = 0) {
		return this.tokens[index] as Token;
	}

	/**
	 * Returns the previous token and then advances the tokens array to the next value.
	 */
	private eat() {
		const prev = this.tokens.shift() as Token;
		return prev;
	}

	/**
	 * Returns the previous token and then advances the tokens array to the next value.
	 *  Also checks the type of expected token and throws if the values dnot match.
	 */
	private expect(type: TokenType, err: any) {
		const prev = this.tokens.shift() as Token;
		if (!prev || prev.type != type) {
            throw Error(`"Parser Error:\n ${err} ${prev} - Expecting: ${type}`)
		}

		return prev;
	}

	public produceAST(sourceCode: string): Program {
		this.tokens = tokenize(sourceCode);
		const program: Program = {
			kind: "Program",
			body: [],
		};

		// Parse until end of file
		while (this.not_eof()) {
			program.body.push(this.parse_stmt());
		}

		return program;
	}

	// Handle complex statement types
	private parse_stmt(): Stmt {
		// skip to parse_expr
		switch (this.at().type) {
			case TokenType.Uint:
			case TokenType.Int:
            case TokenType.Float:
				// Peek further to determine if it's a function or variable declaration
				if (this.at(1).type === TokenType.Identifier) {
					if (this.at(2).type === TokenType.OpenParen) return this.parse_fn_declaration();
					else if (this.at(2).type === TokenType.Equals || this.at(2).type === TokenType.Semicolon) {
						const varDeclaration = this.parse_var_declaration();
						this.expect(TokenType.Semicolon, 'Expect ";" after variable declaration');
						return varDeclaration;
					}
				}
				throw Error("Unexpected token sequence after type specifier");
				// return this.parse_var_declaration();
			case TokenType.Void:
				return this.parse_fn_declaration();
			case TokenType.Global:
				return this.parse_fn_declaration();
			case TokenType.ForLoop:
				return this.parse_forloop_declaration();
			case TokenType.IfCondition:
				return this.parse_if_declaration();
			default:
				const expr = this.parse_expr();
				if (this.at().type === TokenType.Semicolon) this.eat();
				else if (this.at().type !== TokenType.EOF){
					throw Error('Expect Semicolon or EOF after expression')
				}
				// this.expect(TokenType.Semicolon, 'Expect ";" after expression');
				return expr;
		}
	}

	parse_if_declaration(): Stmt {
		this.eat(); // eat if

		const test = this.parse_object_expr();

		this.expect(
			TokenType.OpenBrace,
			"Expected OpenBrace after if"
		);

		const body: Stmt[] = [];

		while (
			this.at().type !== TokenType.EOF &&
			this.at().type !== TokenType.CloseBrace
		) {
			body.push(this.parse_stmt());
		}

		this.expect(
			TokenType.CloseBrace,
			"Closing brace expected inside function declaration"
		);

		const ifDeclaration: ConditionalDeclaration = {
			kind: "ConditionalDeclaration",
			test: test,
			body: body,
		};

		return ifDeclaration;
	}

	parse_forloop_declaration(): Stmt {
		this.eat(); // eat for keyword
		this.expect(TokenType.OpenParen, "Expected open parenthesis");
		
		const init = this.parse_var_declaration();
		this.expect(TokenType.Semicolon, "Expected semicolon after for loop init");

		const condition = this.parse_assignment_expr();
		this.expect(TokenType.Semicolon, "Expected semicolon after for loop");

		const update = this.parse_assignment_expr();
		this.expect(TokenType.CloseParen, "Expected close paren after for loop");

		this.expect(TokenType.OpenBrace, "Expected function body following declaration");
		const body: Stmt[] = [];

		while (
			this.at().type !== TokenType.EOF &&
			this.at().type !== TokenType.CloseBrace
		) {
			body.push(this.parse_stmt());
		}

		this.expect(TokenType.CloseBrace, "Closing brace expected inside function declaration");

		const loop: ForLoopDeclaration = {
			kind: "ForLoopDeclaration",
			init: init,
			condition: condition,
			increment: update,
			body: body
		}

		return loop;
	}

	parse_fn_declaration(): Stmt {
		let qualifiers: string[] = [];
		if (this.at().type === TokenType.Global) {
			qualifiers.push(this.eat().value);
		}

		const type = this.eat(); // eat type keyword
		const name = this.expect(
			TokenType.Identifier,
			"Expected function name following fn keyword"
		).value;

		const args = this.parse_args();

		const params: VarDeclaration[] = [];
		for (const arg of args) {
			if (arg.kind !== "VarDeclaration") {
				console.log(arg);
				throw "Inside function declaration expected parameters to be of type string.";
			}

			params.push(arg as VarDeclaration);
		}

		this.expect(
			TokenType.OpenBrace,
			"Expected function body following declaration"
		);
		const body: Stmt[] = [];

		while (
			this.at().type !== TokenType.EOF &&
			this.at().type !== TokenType.CloseBrace
		) {
			body.push(this.parse_stmt());
		}

		this.expect(
			TokenType.CloseBrace,
			"Closing brace expected inside function declaration"
		);

		const fn: FunctionDeclaration = {
			kind: "FunctionDeclaration",
			name: name,
			qualifiers: qualifiers,
			type: type.value,
			body: body,
			parameters: params,
		};

		return fn;
	}

	// LET IDENT;
	// ( LET | CONST ) IDENT = EXPR;
	parse_var_declaration(): Stmt {
		const varType = this.eat();
		const identifier = this.expect(
			TokenType.Identifier,
			"Expected identifier name following let | const keywords."
		).value;

		if (this.at().type == TokenType.Semicolon) {
			this.eat(); // expect semicolon
			// if (isConstant) {
			// 	throw "Must assigne value to constant expression. No value provided.";
			// }

			return {
				kind: "VarDeclaration",
				type: varType.value,
				identifier,
			} as VarDeclaration;
		}

		this.expect(
			TokenType.Equals,
			"Expected equals token following identifier in var declaration."
		);

		const declaration = {
			kind: "VarDeclaration",
			value: this.parse_expr(),
			type: varType.value,
			identifier,
		} as VarDeclaration;

		// this.expect(
		// 	TokenType.Semicolon,
		// 	"Variable declaration statment must end with semicolon."
		// );

		return declaration;
	}

	// Handle expressions
	private parse_expr(): Expr {
		return this.parse_assignment_expr();
	}

	private parse_assignment_expr(): Expr {
		const left = this.parse_object_expr();

		if (this.at().type == TokenType.Equals) {
			this.eat(); // advance past equals
			const value = this.parse_assignment_expr();
			return { value, assigne: left, kind: "AssignmentExpr" } as AssignmentExpr;
		}

		return left;
	}

	// private parse_assignment_expr(): Expr {
	// 	const left = this.parse_object_expr();

	// 	if (this.at().type == TokenType.Equals) {
	// 		this.eat(); // advance past equals
	// 		const value = this.parse_assignment_expr();
	// 		return { value, assigne: left, kind: "AssignmentExpr" } as AssignmentExpr;
	// 	}

	// 	return left;
	// }

	private parse_object_expr(): Expr {
		if (this.at().type !== TokenType.OpenBrace) {
			return this.parse_compound_expr();
		}

		this.eat(); // advance past open brace.
		const properties = new Array<Property>();

		while (this.not_eof() && this.at().type != TokenType.CloseBrace) {
			const key = this.expect(
				TokenType.Identifier,
				"Object literal key expected"
			).value;

			// Allows shorthand key: pair -> { key, }
			if (this.at().type == TokenType.Comma) {
				this.eat(); // advance past comma
				properties.push({ key, kind: "Property" } as Property);
				continue;
			} // Allows shorthand key: pair -> { key }
			else if (this.at().type == TokenType.CloseBrace) {
				properties.push({ key, kind: "Property" });
				continue;
			}

			// { key: val }
			this.expect(
				TokenType.Colon,
				"Missing colon following identifier in ObjectExpr"
			);
			const value = this.parse_expr();

			properties.push({ kind: "Property", value, key });
			if (this.at().type != TokenType.CloseBrace) {
				this.expect(
					TokenType.Comma,
					"Expected comma or closing bracket following property"
				);
			}
		}

		this.expect(TokenType.CloseBrace, "Object literal missing closing brace.");
		return { kind: "ObjectLiteral", properties } as ObjectLiteral;
	}
	
	// Handle Logical operators 
	private parse_compound_expr(): Expr {
		let left = this.parse_logical_expr();

		while (
			this.at().value == "+=" ||
			this.at().value == "-=" ||
			this.at().value == "*=" ||
			this.at().value == "/="
		) {
			const operator = this.eat().value;
			const right = this.parse_logical_expr();
			left = {
				kind: "CompoundExpr",
				left,
				right,
				operator,
			} as CompoundExpr;
		}

		return left;
	}

	// Handle Logical operators 
	private parse_logical_expr(): Expr {
		let left = this.parse_equality_expr();

		while (
			this.at().value == "&&" ||
			this.at().value == "||"
		) {
			const operator = this.eat().value;
			const right = this.parse_equality_expr();
			left = {
				kind: "LogicalExpr",
				left,
				right,
				operator,
			} as LogicalExpr;
		}

		return left;
	}
	
	// Handle Equality operators 
	private parse_equality_expr(): Expr {
		let left = this.parse_relational_expr();

		while (
			this.at().value == "=="
		) {
			const operator = this.eat().value;
			const right = this.parse_relational_expr();
			left = {
				kind: "EqualityExpr",
				left,
				right,
				operator,
			} as EqualityExpr;
		}

		return left;
	}
	
	// Handle Relational operators 
	private parse_relational_expr(): Expr {
		let left = this.parse_additive_expr();

		while (
			this.at().value == ">" ||
			this.at().value == "<" ||
			this.at().value == "<=" ||
			this.at().value == ">="
		) {
			const operator = this.eat().value;
			const right = this.parse_additive_expr();
			left = {
				kind: "RelationalExpr",
				left,
				right,
				operator,
			} as RelationalExpr;
		}

		return left;
	}

	// Handle Addition & Subtraction Operations
	private parse_additive_expr(): Expr {
		let left = this.parse_multiplicitave_expr();

		while (this.at().value == "+" || this.at().value == "-") {
			const operator = this.eat().value;
			const right = this.parse_multiplicitave_expr();
			left = {
				kind: "BinaryExpr",
				left,
				right,
				operator,
			} as BinaryExpr;
		}

		return left;
	}

	// Handle Multiplication, Division & Modulo Operations
	private parse_multiplicitave_expr(): Expr {
		let left = this.parse_unary_expr();

		while (
			this.at().value == "/" ||
			this.at().value == "*" ||
			this.at().value == "%"
		) {
			const operator = this.eat().value;
			const right = this.parse_unary_expr();
			left = {
				kind: "BinaryExpr",
				left,
				right,
				operator,
			} as BinaryExpr;
		}

		return left;
	}

	private parse_unary_expr(): Expr {
		let left = this.parse_call_member_expr();

		while (
				this.at().value == "++" ||
				this.at().value == "--"
			) {
				
			const operator = this.eat().value;
			left = {
				kind: "UnaryExpr",
				operand: left,
				operator: operator,
			} as UnaryExpr;
		}

		return left;
	}

	// foo.x()()
	private parse_call_member_expr(): Expr {
		const member = this.parse_member_expr();

		if (this.at().type == TokenType.OpenParen) {
			return this.parse_call_expr(member);
		}

		return member;
	}

	private parse_call_expr(caller: Expr): Expr {
		let call_expr: Expr = {
			kind: "CallExpr",
			caller,
			args: this.parse_args(),
		} as CallExpr;

		if (this.at().type == TokenType.OpenParen) {
			call_expr = this.parse_call_expr(call_expr);
		}

		return call_expr;
	}

	private parse_args(): Expr[] {
		this.expect(TokenType.OpenParen, "Expected open parenthesis");
		const args = this.at().type == TokenType.CloseParen ? [] : this.parse_arguments_list();

		this.expect(
			TokenType.CloseParen,
			"Missing closing parenthesis inside arguments list"
		);
		return args;
	}

	private parse_arguments_list(): Expr[] {
		const args = [this.parse_assignment_expr()];

		while (this.at().type == TokenType.Comma && this.eat()) {
			args.push(this.parse_assignment_expr());
		}

		return args;
	}

	private parse_member_expr(): Expr {
		let object = this.parse_primary_expr();

		while (
			this.at().type == TokenType.Dot ||
			this.at().type == TokenType.OpenBracket
		) {
			const operator = this.eat();
			let property: Expr;
			let computed: boolean;

			// non-computed values aka obj.expr
			if (operator.type == TokenType.Dot) {
				computed = false;
				// get identifier
				property = this.parse_primary_expr();
				if (property.kind != "Identifier") {
					throw `Cannot use dot operator without right hand side being a identifier`;
				}
			} else {
				// this allows obj[computedValue]
				computed = true;
				property = this.parse_expr();
				this.expect(
					TokenType.CloseBracket,
					"Missing closing bracket in computed value."
				);
			}

			object = {
				kind: "MemberExpr",
				object,
				property,
				computed,
			} as MemberExpr;
		}

		return object;
	}

	// Orders Of Prescidence
	// Assignment
	// Object
	// AdditiveExpr
	// MultiplicitaveExpr
	// Call
	// Member
	// PrimaryExpr

	// Parse Literal Values & Grouping Expressions
	private parse_primary_expr(): Expr {
		const tk = this.at().type;

		// Determine which token we are currently at and return literal value
		switch (tk) {
			// User defined values.
			case TokenType.Identifier:
				return { kind: "Identifier", symbol: this.eat().value } as Identifier;

			// Constants and Numeric Constants
			case TokenType.Number:
				let num = this.eat().value;
				if (this.at().type === TokenType.Dot) num += this.eat().value;
				// Should support float a = 10.;?
				if (this.at().type === TokenType.Number) num += this.eat().value;
				return {
					kind: "NumericLiteral",
					value: parseFloat(num),
				} as NumericLiteral;

			case TokenType.Uint:
			case TokenType.Int:
			case TokenType.Float:
				const type = this.eat().value;
				const isPointer = this.at().type === TokenType.Asterisk ? true : false;
				if (isPointer) this.eat();
				const varDeclaration: VarDeclaration = {
					kind: "VarDeclaration",
					type: type,
					pointer: isPointer,
					identifier: this.eat().value,
				};
				return varDeclaration;

			// Grouping Expressions
			case TokenType.OpenParen: {
				this.eat(); // eat the opening paren
				const value = this.parse_expr();
				this.expect(
					TokenType.CloseParen,
					"Unexpected token found inside parenthesised expression. Expected closing parenthesis."
				); // closing paren
				return value;
			}

			// Unidentified Tokens and Invalid Code Reached
			default:
				console.error("Unexpected token found during parsing!", this.at());
                throw Error("Error parsing primary expression");
		}
	}
}