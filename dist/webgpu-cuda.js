var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};

// src/Lexer.ts
var KEYWORDS = {
  int: 2 /* Int */,
  uint: 3 /* Uint */,
  float: 4 /* Float */,
  void: 34 /* Void */,
  for: 30 /* ForLoop */,
  while: 31 /* WhileLoop */,
  if: 32 /* IfCondition */,
  "#define": 35 /* Define */,
  __global__: 39 /* Global */,
  __shared__: 37 /* Shared */,
  extern: 36 /* Extern */,
  const: 38 /* Const */
};
function token(value = "", type) {
  return { value, type };
}
function isword(src) {
  return /^[a-zA-Z0-9_#]+$/.test(src);
}
function isskippable(str) {
  return str == " " || str == "\n" || str == "	" || str == "\r";
}
function isint(str) {
  const c = str.charCodeAt(0);
  const bounds = ["0".charCodeAt(0), "9".charCodeAt(0)];
  return c >= bounds[0] && c <= bounds[1];
}
function tokenize(sourceCode) {
  const tokens = new Array();
  const src = sourceCode.split("");
  while (src.length > 0) {
    if (src[0] == "/") {
      const isComment = src[1] === "/" || src[1] === "*";
      const commentType = isComment && src[1] === "/" ? "single" : "multi";
      if (isComment) {
        let comment = "";
        while (src.length > 0) {
          comment += src.shift();
          if (commentType === "single" && comment[comment.length - 1] === "\n")
            break;
          else if (commentType === "multi" && comment[comment.length - 2] === "*" && comment[comment.length - 1] === "/")
            break;
        }
        if (commentType == "multi")
          comment.slice(0, comment.length - 4);
        tokens.push(token(comment, 40 /* Comment */));
      } else {
        tokens.push(token(src.shift(), 33 /* Asterisk */));
      }
    } else if (src[0] == "(")
      tokens.push(token(src.shift(), 19 /* OpenParen */));
    else if (src[0] == ")")
      tokens.push(token(src.shift(), 20 /* CloseParen */));
    else if (src[0] == "{")
      tokens.push(token(src.shift(), 21 /* OpenBrace */));
    else if (src[0] == "}")
      tokens.push(token(src.shift(), 22 /* CloseBrace */));
    else if (src[0] == "[")
      tokens.push(token(src.shift(), 23 /* OpenBracket */));
    else if (src[0] == "]")
      tokens.push(token(src.shift(), 24 /* CloseBracket */));
    else if (src[0] == "+" && src[1] && src[1] === "+")
      tokens.push(token(`${src.shift()}${src.shift()}`, 6 /* UnaryOperator */));
    else if (src[0] == "-" && src[1] && src[1] === "-")
      tokens.push(token(`${src.shift()}${src.shift()}`, 6 /* UnaryOperator */));
    else if (src[0] == ">" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 8 /* RelationalOperator */));
    else if (src[0] == "<" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 8 /* RelationalOperator */));
    else if (src[0] == "=" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 9 /* EqualityOperator */));
    else if (src[0] == "&" && src[1] && src[1] === "&")
      tokens.push(token(`${src.shift()}${src.shift()}`, 10 /* LogicalOperator */));
    else if (src[0] == "|" && src[1] && src[1] === "|")
      tokens.push(token(`${src.shift()}${src.shift()}`, 10 /* LogicalOperator */));
    else if (src[0] == "+" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 11 /* CompoundOperator */));
    else if (src[0] == "-" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 11 /* CompoundOperator */));
    else if (src[0] == "*" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 11 /* CompoundOperator */));
    else if (src[0] == "/" && src[1] && src[1] === "=")
      tokens.push(token(`${src.shift()}${src.shift()}`, 11 /* CompoundOperator */));
    else if (src[0] == "+" || src[0] == "-" || src[0] == "/" || src[0] == "%") {
      tokens.push(token(src.shift(), 7 /* BinaryOperator */));
    } else if (src[0] == "*")
      tokens.push(token(src.shift(), 33 /* Asterisk */));
    else if (src[0] == "=")
      tokens.push(token(src.shift(), 12 /* Equals */));
    else if (src[0] == ">")
      tokens.push(token(src.shift(), 8 /* RelationalOperator */));
    else if (src[0] == "<")
      tokens.push(token(src.shift(), 8 /* RelationalOperator */));
    else if (src[0] == ";")
      tokens.push(token(src.shift(), 18 /* Semicolon */));
    else if (src[0] == ":")
      tokens.push(token(src.shift(), 17 /* Colon */));
    else if (src[0] == ",")
      tokens.push(token(src.shift(), 15 /* Comma */));
    else if (src[0] == ".")
      tokens.push(token(src.shift(), 16 /* Dot */));
    else if (src[0] == "&")
      tokens.push(token(src.shift(), 13 /* Ampersand */));
    else if (src[0] == "|")
      tokens.push(token(src.shift(), 14 /* Pipe */));
    else {
      if (isint(src[0])) {
        let num = "";
        while (src.length > 0 && isint(src[0])) {
          num += src.shift();
        }
        tokens.push(token(num, 0 /* Number */));
      } else if (isword(src[0])) {
        let ident = "";
        while (src.length > 0 && isword(src[0])) {
          ident += src.shift();
        }
        const reserved = KEYWORDS[ident];
        if (typeof reserved == "number")
          tokens.push(token(ident, reserved));
        else
          tokens.push(token(ident, 1 /* Identifier */));
      } else if (isskippable(src[0])) {
        src.shift();
      } else {
        throw Error(`Unreconized character found in source: ${src[0].charCodeAt(0)} ${src[0]}`);
      }
    }
  }
  tokens.push({ type: 42 /* EOF */, value: "EndOfFile" });
  return tokens;
}

// src/Parser.ts
var Parser = class {
  constructor() {
    __publicField(this, "tokens", []);
  }
  /*
   * Determines if the parsing is complete and the END OF FILE Is reached.
   */
  not_eof() {
    return this.tokens[0].type != 42 /* EOF */;
  }
  /**
   * Returns the currently available token
   */
  at(index = 0) {
    return this.tokens[index];
  }
  /**
   * Returns the previous token and then advances the tokens array to the next value.
   */
  eat() {
    const prev = this.tokens.shift();
    return prev;
  }
  /**
   * Returns the previous token and then advances the tokens array to the next value.
   *  Also checks the type of expected token and throws if the values dnot match.
   */
  expect(type, err) {
    const prev = this.tokens.shift();
    if (!prev || prev.type != type) {
      throw Error(`"Parser Error:
 ${err} ${prev} - Expecting: ${type}`);
    }
    return prev;
  }
  produceAST(sourceCode) {
    this.tokens = tokenize(sourceCode);
    const program = {
      kind: "Program",
      body: []
    };
    while (this.not_eof()) {
      program.body.push(this.parse_stmt());
    }
    return program;
  }
  // Handle complex statement types
  parse_stmt() {
    switch (this.at().type) {
      case 3 /* Uint */:
      case 2 /* Int */:
      case 4 /* Float */:
        if (this.at(1).type === 1 /* Identifier */) {
          if (this.at(2).type === 19 /* OpenParen */)
            return this.parse_fn_declaration();
          else if (this.at(2).type === 12 /* Equals */ || this.at(2).type === 18 /* Semicolon */) {
            const varDeclaration = this.parse_var_declaration();
            this.expect(18 /* Semicolon */, 'Expect ";" after variable declaration');
            return varDeclaration;
          }
        }
        throw Error("Unexpected token sequence after type specifier");
      case 34 /* Void */:
        return this.parse_fn_declaration();
      case 39 /* Global */:
        return this.parse_fn_declaration();
      case 30 /* ForLoop */:
        return this.parse_forloop_declaration();
      case 32 /* IfCondition */:
        return this.parse_if_declaration();
      default:
        const expr = this.parse_expr();
        if (this.at().type === 18 /* Semicolon */)
          this.eat();
        else if (this.at().type !== 42 /* EOF */) {
          throw Error("Expect Semicolon or EOF after expression");
        }
        return expr;
    }
  }
  parse_if_declaration() {
    this.eat();
    const test = this.parse_object_expr();
    this.expect(
      21 /* OpenBrace */,
      "Expected OpenBrace after if"
    );
    const body = [];
    while (this.at().type !== 42 /* EOF */ && this.at().type !== 22 /* CloseBrace */) {
      body.push(this.parse_stmt());
    }
    this.expect(
      22 /* CloseBrace */,
      "Closing brace expected inside function declaration"
    );
    const ifDeclaration = {
      kind: "ConditionalDeclaration",
      test,
      body
    };
    return ifDeclaration;
  }
  parse_forloop_declaration() {
    this.eat();
    this.expect(19 /* OpenParen */, "Expected open parenthesis");
    const init = this.parse_var_declaration();
    this.expect(18 /* Semicolon */, "Expected semicolon after for loop init");
    const condition = this.parse_assignment_expr();
    this.expect(18 /* Semicolon */, "Expected semicolon after for loop");
    const update = this.parse_assignment_expr();
    this.expect(20 /* CloseParen */, "Expected close paren after for loop");
    this.expect(21 /* OpenBrace */, "Expected function body following declaration");
    const body = [];
    while (this.at().type !== 42 /* EOF */ && this.at().type !== 22 /* CloseBrace */) {
      body.push(this.parse_stmt());
    }
    this.expect(22 /* CloseBrace */, "Closing brace expected inside function declaration");
    const loop = {
      kind: "ForLoopDeclaration",
      init,
      condition,
      increment: update,
      body
    };
    return loop;
  }
  parse_fn_declaration() {
    let qualifiers = [];
    if (this.at().type === 39 /* Global */) {
      qualifiers.push(this.eat().value);
    }
    const type = this.eat();
    const name = this.expect(
      1 /* Identifier */,
      "Expected function name following fn keyword"
    ).value;
    const args = this.parse_args();
    const params = [];
    for (const arg of args) {
      if (arg.kind !== "VarDeclaration") {
        console.log(arg);
        throw "Inside function declaration expected parameters to be of type string.";
      }
      params.push(arg);
    }
    this.expect(
      21 /* OpenBrace */,
      "Expected function body following declaration"
    );
    const body = [];
    while (this.at().type !== 42 /* EOF */ && this.at().type !== 22 /* CloseBrace */) {
      body.push(this.parse_stmt());
    }
    this.expect(
      22 /* CloseBrace */,
      "Closing brace expected inside function declaration"
    );
    const fn = {
      kind: "FunctionDeclaration",
      name,
      qualifiers,
      type: type.value,
      body,
      parameters: params
    };
    return fn;
  }
  // LET IDENT;
  // ( LET | CONST ) IDENT = EXPR;
  parse_var_declaration() {
    const varType = this.eat();
    const identifier = this.expect(
      1 /* Identifier */,
      "Expected identifier name following let | const keywords."
    ).value;
    if (this.at().type == 18 /* Semicolon */) {
      this.eat();
      return {
        kind: "VarDeclaration",
        type: varType.value,
        identifier
      };
    }
    this.expect(
      12 /* Equals */,
      "Expected equals token following identifier in var declaration."
    );
    const declaration = {
      kind: "VarDeclaration",
      value: this.parse_expr(),
      type: varType.value,
      identifier
    };
    return declaration;
  }
  // Handle expressions
  parse_expr() {
    return this.parse_assignment_expr();
  }
  parse_assignment_expr() {
    const left = this.parse_object_expr();
    if (this.at().type == 12 /* Equals */) {
      this.eat();
      const value = this.parse_assignment_expr();
      return { value, assigne: left, kind: "AssignmentExpr" };
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
  parse_object_expr() {
    if (this.at().type !== 21 /* OpenBrace */) {
      return this.parse_compound_expr();
    }
    this.eat();
    const properties = new Array();
    while (this.not_eof() && this.at().type != 22 /* CloseBrace */) {
      const key = this.expect(
        1 /* Identifier */,
        "Object literal key expected"
      ).value;
      if (this.at().type == 15 /* Comma */) {
        this.eat();
        properties.push({ key, kind: "Property" });
        continue;
      } else if (this.at().type == 22 /* CloseBrace */) {
        properties.push({ key, kind: "Property" });
        continue;
      }
      this.expect(
        17 /* Colon */,
        "Missing colon following identifier in ObjectExpr"
      );
      const value = this.parse_expr();
      properties.push({ kind: "Property", value, key });
      if (this.at().type != 22 /* CloseBrace */) {
        this.expect(
          15 /* Comma */,
          "Expected comma or closing bracket following property"
        );
      }
    }
    this.expect(22 /* CloseBrace */, "Object literal missing closing brace.");
    return { kind: "ObjectLiteral", properties };
  }
  // Handle Logical operators 
  parse_compound_expr() {
    let left = this.parse_logical_expr();
    while (this.at().value == "+=" || this.at().value == "-=" || this.at().value == "*=" || this.at().value == "/=") {
      const operator = this.eat().value;
      const right = this.parse_logical_expr();
      left = {
        kind: "CompoundExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  // Handle Logical operators 
  parse_logical_expr() {
    let left = this.parse_equality_expr();
    while (this.at().value == "&&" || this.at().value == "||") {
      const operator = this.eat().value;
      const right = this.parse_equality_expr();
      left = {
        kind: "LogicalExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  // Handle Equality operators 
  parse_equality_expr() {
    let left = this.parse_relational_expr();
    while (this.at().value == "==") {
      const operator = this.eat().value;
      const right = this.parse_relational_expr();
      left = {
        kind: "EqualityExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  // Handle Relational operators 
  parse_relational_expr() {
    let left = this.parse_additive_expr();
    while (this.at().value == ">" || this.at().value == "<" || this.at().value == "<=" || this.at().value == ">=") {
      const operator = this.eat().value;
      const right = this.parse_additive_expr();
      left = {
        kind: "RelationalExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  // Handle Addition & Subtraction Operations
  parse_additive_expr() {
    let left = this.parse_multiplicitave_expr();
    while (this.at().value == "+" || this.at().value == "-") {
      const operator = this.eat().value;
      const right = this.parse_multiplicitave_expr();
      left = {
        kind: "BinaryExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  // Handle Multiplication, Division & Modulo Operations
  parse_multiplicitave_expr() {
    let left = this.parse_unary_expr();
    while (this.at().value == "/" || this.at().value == "*" || this.at().value == "%") {
      const operator = this.eat().value;
      const right = this.parse_unary_expr();
      left = {
        kind: "BinaryExpr",
        left,
        right,
        operator
      };
    }
    return left;
  }
  parse_unary_expr() {
    let left = this.parse_call_member_expr();
    while (this.at().value == "++" || this.at().value == "--") {
      const operator = this.eat().value;
      left = {
        kind: "UnaryExpr",
        operand: left,
        operator
      };
    }
    return left;
  }
  // foo.x()()
  parse_call_member_expr() {
    const member = this.parse_member_expr();
    if (this.at().type == 19 /* OpenParen */) {
      return this.parse_call_expr(member);
    }
    return member;
  }
  parse_call_expr(caller) {
    let call_expr = {
      kind: "CallExpr",
      caller,
      args: this.parse_args()
    };
    if (this.at().type == 19 /* OpenParen */) {
      call_expr = this.parse_call_expr(call_expr);
    }
    return call_expr;
  }
  parse_args() {
    this.expect(19 /* OpenParen */, "Expected open parenthesis");
    const args = this.at().type == 20 /* CloseParen */ ? [] : this.parse_arguments_list();
    this.expect(
      20 /* CloseParen */,
      "Missing closing parenthesis inside arguments list"
    );
    return args;
  }
  parse_arguments_list() {
    const args = [this.parse_assignment_expr()];
    while (this.at().type == 15 /* Comma */ && this.eat()) {
      args.push(this.parse_assignment_expr());
    }
    return args;
  }
  parse_member_expr() {
    let object = this.parse_primary_expr();
    while (this.at().type == 16 /* Dot */ || this.at().type == 23 /* OpenBracket */) {
      const operator = this.eat();
      let property;
      let computed;
      if (operator.type == 16 /* Dot */) {
        computed = false;
        property = this.parse_primary_expr();
        if (property.kind != "Identifier") {
          throw `Cannot use dot operator without right hand side being a identifier`;
        }
      } else {
        computed = true;
        property = this.parse_expr();
        this.expect(
          24 /* CloseBracket */,
          "Missing closing bracket in computed value."
        );
      }
      object = {
        kind: "MemberExpr",
        object,
        property,
        computed
      };
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
  parse_primary_expr() {
    const tk = this.at().type;
    switch (tk) {
      case 1 /* Identifier */:
        return { kind: "Identifier", symbol: this.eat().value };
      case 0 /* Number */:
        let num = this.eat().value;
        if (this.at().type === 16 /* Dot */)
          num += this.eat().value;
        if (this.at().type === 0 /* Number */)
          num += this.eat().value;
        return {
          kind: "NumericLiteral",
          value: parseFloat(num)
        };
      case 3 /* Uint */:
      case 2 /* Int */:
      case 4 /* Float */:
        const type = this.eat().value;
        const isPointer = this.at().type === 33 /* Asterisk */ ? true : false;
        if (isPointer)
          this.eat();
        const varDeclaration = {
          kind: "VarDeclaration",
          type,
          pointer: isPointer,
          identifier: this.eat().value
        };
        return varDeclaration;
      case 19 /* OpenParen */: {
        this.eat();
        const value = this.parse_expr();
        this.expect(
          20 /* CloseParen */,
          "Unexpected token found inside parenthesised expression. Expected closing parenthesis."
        );
        return value;
      }
      default:
        console.error("Unexpected token found during parsing!", this.at());
        throw Error("Error parsing primary expression");
    }
  }
};

// src/CodeGenerator.ts
var CudaTypeToWGSL = {
  int: "i32",
  uint: "u32",
  float: "f32",
  void: ""
};
function getPrecedence(operator) {
  const precedence = {
    "*": 3,
    "/": 3,
    "%": 3,
    "+": 2,
    "-": 2,
    "<": 1,
    "<=": 1,
    ">": 1,
    ">=": 1,
    "==": 0,
    "!=": 0
    // add other operators and their precedence as necessary
  };
  return precedence[operator] || -1;
}
function needsParentheses(parentOp, childOp, isRight) {
  const parentPrec = getPrecedence(parentOp);
  const childPrec = getPrecedence(childOp);
  if (childPrec < parentPrec)
    return true;
  if (childPrec === parentPrec && isRight)
    return true;
  return false;
}
function codeGenerator(node) {
  switch (node.kind) {
    case "Program":
      const program = node;
      const globalCode = program.globals.map(codeGenerator).join("\n");
      const bodyCode = program.body.map(codeGenerator).join("\n");
      return `${globalCode}
${bodyCode}`;
    case "Identifier":
      const identifier = node;
      return identifier.symbol;
    case "NumericLiteral":
      const numericLiteral = node;
      return numericLiteral.value;
    case "BinaryExpr": {
      const binaryExpression = node;
      let left = codeGenerator(binaryExpression.left);
      let right = codeGenerator(binaryExpression.right);
      if (binaryExpression.left.kind === "BinaryExpr" && needsParentheses(binaryExpression.operator, binaryExpression.left.operator, false)) {
        left = `(${left})`;
      }
      if (binaryExpression.right.kind === "BinaryExpr" && needsParentheses(binaryExpression.operator, binaryExpression.right.operator, true)) {
        right = `(${right})`;
      }
      return `${left} ${binaryExpression.operator} ${right}`;
    }
    case "RelationalExpr":
    case "EqualityExpr":
    case "LogicalExpr": {
      const binaryExpression = node;
      const left = codeGenerator(binaryExpression.left);
      const right = codeGenerator(binaryExpression.right);
      return `${left} ${binaryExpression.operator} ${right}`;
    }
    case "UnaryExpr":
      const unaryExpr = node;
      return `${codeGenerator(unaryExpr.operand)}${unaryExpr.operator}`;
    case "VarDeclaration":
      const varDeclaration = node;
      const semicolon = varDeclaration.value && varDeclaration.value.kind !== "CallExpr" ? ";" : "";
      if (varDeclaration.value) {
        return `var ${varDeclaration.identifier}: ${CudaTypeToWGSL[varDeclaration.type]} = ${codeGenerator(varDeclaration.value)}${semicolon}`;
      } else {
        if (varDeclaration.pointer)
          return `${varDeclaration.identifier}: array<${CudaTypeToWGSL[varDeclaration.type]}>`;
        else
          return `${varDeclaration.identifier}: ${CudaTypeToWGSL[varDeclaration.type]}`;
      }
    case "CompoundExpr": {
      const compoundExpr = node;
      const left = codeGenerator(compoundExpr.left);
      const right = codeGenerator(compoundExpr.right);
      return `${left} ${compoundExpr.operator} ${right};`;
    }
    case "AssignmentExpr": {
      const assignmentExpr = node;
      return `${codeGenerator(assignmentExpr.assigne)} = ${codeGenerator(assignmentExpr.value)};`;
    }
    case "ForLoopDeclaration": {
      const forLoopDeclaration = node;
      const init = `${codeGenerator(forLoopDeclaration.init)}`;
      const condition = `${codeGenerator(forLoopDeclaration.condition)}`;
      const increment = `${codeGenerator(forLoopDeclaration.increment)}`;
      const body = `${forLoopDeclaration.body.map(codeGenerator).join("\n")}`;
      return `for (${init} ${condition}; ${increment}) {
${body}
}`;
    }
    case "ConditionalDeclaration": {
      const conditionalDeclaration = node;
      const test = codeGenerator(conditionalDeclaration.test);
      const body = conditionalDeclaration.body.map(codeGenerator).join("\n");
      return `if (${test}) {
${body}
}`;
    }
    case "MemberExpr":
      const memberExpr = node;
      if (memberExpr.computed)
        return `${codeGenerator(memberExpr.object)}[${codeGenerator(memberExpr.property)}]`;
      else
        return `${codeGenerator(memberExpr.object)}.${codeGenerator(memberExpr.property)}`;
    case "CallExpr":
      const callExpr = node;
      return `${codeGenerator(callExpr.caller)}(${callExpr.args.map(codeGenerator).join(", ")});`;
    case "FunctionDeclaration": {
      const functionDeclaration = node;
      let body = functionDeclaration.body.map(codeGenerator).join("\n");
      if (functionDeclaration.body.length > 0)
        body = "\n" + body + "\n";
      let returnType = "";
      if (functionDeclaration.type !== "void")
        returnType = ` -> ${CudaTypeToWGSL[functionDeclaration.type]}`;
      const args = functionDeclaration.parameters.map(codeGenerator).join(", ");
      return `fn ${functionDeclaration.name}(${args})${returnType} {${body}}`;
    }
    case "BufferDeclaration": {
      const bufferDeclaration = node;
      const group = `@group(%group_${bufferDeclaration.bufferCounter}%)`;
      const binding = `@binding(%binding_${bufferDeclaration.bufferCounter}%)`;
      const param = bufferDeclaration.value;
      const paramStr = param.pointer ? `array<${CudaTypeToWGSL[param.type]}, %array_size_${bufferDeclaration.bufferCounter}%>` : `${CudaTypeToWGSL[param.type]}`;
      return `${group} ${binding} var<${bufferDeclaration.storage}, ${bufferDeclaration.access}> ${bufferDeclaration.identifier}: ${paramStr};`;
    }
    default:
      console.warn("[CodeGenerator] Node handler not found", node);
      throw new Error(`Cannot handle ${node.kind}`);
  }
}

// src/Traverser.ts
function traverser(ast, visitor) {
  function traverseArray(array, parent) {
    array.forEach((child) => {
      traverseNode(child, parent);
    });
  }
  function traverseNode(node, parent) {
    let methods = visitor[node.kind];
    if (methods && methods.enter)
      methods.enter(node, parent);
    else {
      console.warn(node);
      throw Error("Enter method not defined");
    }
    switch (node.kind) {
      case "Program":
        const program = node;
        traverseArray(program.body, node);
        break;
      case "UnaryExpr":
        const unaryExpr = node;
        traverseNode(unaryExpr.operand, node);
        break;
      case "EqualityExpr":
      case "LogicalExpr":
      case "RelationalExpr":
      case "CompoundExpr":
      case "BinaryExpr":
        const binaryExpr = node;
        traverseNode(binaryExpr.left, node);
        traverseNode(binaryExpr.right, node);
        break;
      case "CallExpr":
        const callExpr = node;
        traverseArray(callExpr.args, node);
        break;
      case "AssignmentExpr":
        const assignmentExpr = node;
        traverseNode(assignmentExpr.assigne, node);
        traverseNode(assignmentExpr.value, node);
        break;
      case "NumericLiteral":
      case "Identifier":
        break;
      case "ForLoopDeclaration":
        const forLoopDeclaration = node;
        traverseNode(forLoopDeclaration.init, node);
        traverseNode(forLoopDeclaration.condition, node);
        traverseNode(forLoopDeclaration.increment, node);
        traverseArray(forLoopDeclaration.body, node);
        break;
      case "ConditionalDeclaration":
        const conditionalDeclaration = node;
        traverseNode(conditionalDeclaration.test, node);
        traverseArray(conditionalDeclaration.body, node);
        break;
      case "VarDeclaration":
        const varDeclaration = node;
        if (varDeclaration.value)
          traverseNode(varDeclaration.value, node);
        break;
      case "MemberExpr":
        const memberExpr = node;
        traverseNode(memberExpr.object, node);
        traverseNode(memberExpr.property, node);
        break;
      case "FunctionDeclaration":
        const functionDeclaration = node;
        traverseArray(functionDeclaration.parameters, node);
        traverseArray(functionDeclaration.body, node);
        break;
      case "BufferDeclaration":
        console.log('case "BufferDeclaration":', node);
        throw Error("ERGERGERG");
      default:
        console.warn("[Traverser] Error processing node", node);
        throw new Error(`Cannot handle ${node.kind}`);
    }
    if (methods && methods.exit) {
      methods.exit(node, parent);
    }
  }
  traverseNode(ast, null);
}

// src/Transformer.ts
function transformer(ast) {
  let newAst = {
    kind: "Program",
    body: [],
    globals: [],
    counters: {
      buffer: 0
    }
  };
  ast._context = {
    body: newAst.body,
    globals: newAst.globals,
    counters: newAst.counters
  };
  traverser(ast, {
    Program: {
      enter(node, parent) {
      }
    },
    NumericLiteral: {
      // We'll visit them on enter.
      enter(node, parent) {
        parent._context?.body.push({
          kind: "NumericLiteral",
          value: node.value
        });
      }
    },
    BinaryExpr: {
      enter(node, parent) {
        node._context = { body: [node.left, node.right], globals: [] };
        parent._context?.body.push(node);
      }
    },
    RelationalExpr: {
      enter(node, parent) {
        node._context = { body: [node.left, node.right], globals: [] };
        parent._context?.body.push(node);
      }
    },
    EqualityExpr: {
      enter(node, parent) {
        node._context = { body: [node.left, node.right], globals: [] };
        parent._context?.body.push(node);
      }
    },
    LogicalExpr: {
      enter(node, parent) {
        node._context = { body: [node.left, node.right], globals: [] };
        parent._context?.body.push(node);
      }
    },
    UnaryExpr: {
      enter(node, parent) {
        node._context = { body: [node.operand], globals: [] };
        parent._context?.body.push(node);
      }
    },
    CompoundExpr: {
      enter(node, parent) {
        node._context = { body: [node.left, node.right], globals: [] };
        parent._context?.body.push(node);
      }
    },
    AssignmentExpr: {
      enter(node, parent) {
        node._context = { body: [node.assigne, node.value], globals: [] };
        parent._context?.body.push(node);
      }
    },
    ForLoopDeclaration: {
      enter(node, parent) {
        node._context = { body: [node.init, node.condition, node.increment, ...node.body], globals: [] };
        parent._context?.body.push(node);
      }
    },
    ConditionalDeclaration: {
      enter(node, parent) {
        node._context = {
          body: [node.test, ...node.body],
          globals: []
        };
        parent._context?.body.push(node);
      }
    },
    VarDeclaration: {
      enter(node, parent) {
        node._context = { body: [node.value], globals: [] };
        parent._context?.body.push(node);
      }
    },
    MemberExpr: {
      enter(node, parent) {
        node._context = {
          body: [node.object, node.property],
          globals: []
        };
        parent._context?.body.push(node);
      }
    },
    Identifier: {
      enter(node, parent) {
        parent._context?.body.push(node);
      }
    },
    CallExpr: {
      enter(node, parent) {
        const expression = {
          kind: "CallExpr",
          args: [],
          caller: node.caller
        };
        node._context = { body: expression.args, globals: [] };
        parent._context?.body.push(expression);
      }
    },
    BufferDeclaration: {
      enter(node, parent) {
      }
    },
    FunctionDeclaration: {
      enter(node, parent) {
        const func = {
          kind: "FunctionDeclaration",
          name: node.name,
          qualifiers: node.qualifiers,
          type: node.type,
          body: node.body,
          parameters: []
        };
        const body = [...func.body];
        if (!node.qualifiers.includes("__global__")) {
          func.parameters.push(...node.parameters);
          body.push(...func.parameters);
        }
        const globals = [];
        if (node.qualifiers.includes("__global__")) {
          for (let param of node.parameters) {
            const bufferDeclaration = {
              kind: "BufferDeclaration",
              bufferCounter: parent._context.counters.buffer++,
              storage: "storage",
              access: "read_write",
              identifier: param.identifier,
              value: param
            };
            globals.push(bufferDeclaration);
          }
        }
        node._context = {
          body,
          globals
        };
        parent._context?.globals.push(...globals);
        parent._context?.body.push(func);
      }
    }
  });
  return newAst;
}
export {
  Parser,
  codeGenerator,
  transformer
};
